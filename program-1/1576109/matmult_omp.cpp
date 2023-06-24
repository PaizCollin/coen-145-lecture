#include <assert.h>     // assert
#include <errno.h>      // errno
#include <stdio.h>      // fopen, fscanf, fprintf, fclose
#include <stdlib.h>     // EXIT_SUCCESS, EXIT_FAILURE, malloc, free
#include <omp.h>        // #pragma omp
#include <fstream>      // ofstream, open, close
#include <iostream>     // cout

static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/

    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

static int mult_mat(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t i, j, k;
  double sum;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  for (i=0; i<n; ++i) {
    for (j=0; j<p; ++j) {
      for (k=0, sum=0.0; k<m; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      C[i*p+j] = sum;
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

static int mult_mat_block(size_t const n, size_t const m, size_t const p,
                          double const * const A, double const * const B,
                          double ** const Cp, size_t blocksize)
{
  size_t nstart, nend, i, j, k;
  double sum;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }

  // parallelize each block
  // shared result array C
  // private iteration variables nend, i, j, k, sum
  #pragma omp parallel for shared(C) private(nend, i, j, k, sum)
  for (nstart=0; nstart<n; nstart+=blocksize) {
    nend = nstart + blocksize - 1;
    if(nend >= n)
      nend = n-1;
    for (i=nstart; i<=nend; ++i) {
      for (j=0; j<p; j++) {
            sum = 0.0;
        for (k=0; k<m; k++) {
          sum += A[i*m+k] * B[k*p+j];
        }
        C[i*p+j] += sum;
      }  
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

static int mult_mat_tiling(size_t const n, size_t const m, size_t const p,
                          double const * const A, double const * const B,
                          double ** const Cp, size_t nts, size_t mts, size_t pts)
{
  size_t nstart, nend, mstart, mend, pstart, pend, i, j, k;
  double sum;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }


  // parallelize each tile
  // shared result array C
  // private iteration variables pstart, mstart, nend, pend, mend, i, j, k, sum
  #pragma omp parallel for shared(C) private(pstart, mstart, nend, pend, mend, i, j, k, sum)
  for (nstart=0; nstart<n; nstart+=nts) {
    nend = nstart + nts - 1;
    if(nend >= n)
      nend = n-1;
    for (pstart=0; pstart<p; pstart+=pts) {
      pend = pstart + pts - 1;
      if(pend >= p)
        pend = p-1;
      for (mstart=0; mstart<m; mstart+=mts) {
        mend = mstart + mts - 1;
        if(mend >= m)
          mend = m-1;
        for (i=nstart; i<=nend; ++i) {
          for (j=pstart; j<=pend; ++j) {
            sum = 0.0;     
            for (k=mstart; k<=mend; ++k) {
              sum += A[i*m+k] * B[k*p+j];
            }
            C[i*p+j] += sum;
          }
        }
      }  
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}


int main(int argc, char * argv[])
{
  size_t nrows, ncols, ncols2, nthreads, tilesize, blocksize;
  double * A=NULL, * B=NULL, * C=NULL, * D=NULL, * E=NULL;
  double serialStart, serialEnd, serialTime, blockStart, blockEnd, blockTime, tilingStart, tilingEnd, tilingTime;

  /** open files to append results **/
  std::ofstream serialFile;
	serialFile.open("serial.csv", std::ofstream::app);
  std::ofstream blockFile;
	blockFile.open("block.csv", std::ofstream::app);
  std::ofstream tilingFile;
	tilingFile.open("tiling.csv", std::ofstream::app);

  /** check for command line arguments **/
  if (argc != 7) {
    fprintf(stderr, "usage: matmult nrows ncols ncols2 nthreads tilesize blocksize\n");
    goto failure;
  }

  /** set command line argument values to variables **/
  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  ncols2 = atoi(argv[3]);
  nthreads = atoi(argv[4]);
  tilesize = atoi(argv[5]);
  blocksize = atoi(argv[6]);

  /** create matrix A **/
  if (create_mat(nrows, ncols, &A)) {
    perror("error");
    goto failure;
  }

  /** create matrix B **/
  if (create_mat(ncols, ncols2, &B)) {
    perror("error");
    goto failure;
  }

  /** set thread count **/
  omp_set_num_threads(nthreads);

  /** serial code start (only run once per matrix size change) **/
  if(nthreads == 1) {
    serialStart = omp_get_wtime();
    if (mult_mat(nrows, ncols, ncols2, A, B, &C)) {
      perror("error");
      goto failure;
    }
    serialEnd = omp_get_wtime();
    serialTime = serialEnd - serialStart;
  }
  /** serial code end **/

  /** block strategy start **/
  blockStart = omp_get_wtime();
  if (mult_mat_block(nrows, ncols, ncols2, A, B, &D, blocksize)) {
    perror("error");
    goto failure;
  }
  blockEnd = omp_get_wtime();
	blockTime = blockEnd - blockStart;
  /** block strategy end **/

  /** tiling strategy start **/
  tilingStart = omp_get_wtime();
  if (mult_mat_tiling(nrows, ncols, ncols2, A, B, &E, tilesize, tilesize, tilesize)) {
    perror("error");
    goto failure;
  }
  tilingEnd = omp_get_wtime();
	tilingTime = tilingEnd - tilingStart;
  /** tiling strategy end **/

  /** report serial results to serial.csv **/
  if(nthreads == 1) {
    serialFile << serialTime << std::endl;
    serialFile.close(); 
  }

  /** report block results to blocking.csv **/
  blockFile << nthreads << " " << blocksize << " " << blockTime << std::endl;
	blockFile.close();

  /** report tiling results to tiling.csv **/
  tilingFile << nthreads << " " << tilesize << " " << tilingTime << std::endl;
	tilingFile.close();

  /** free memory **/
  free(A);
  free(B);
  free(C);

  return EXIT_SUCCESS;

  failure:
  if(A){
    free(A);
  }
  if(B){
    free(B);
  }
  if(C){
    free(C);
  }

  return EXIT_FAILURE;
}
