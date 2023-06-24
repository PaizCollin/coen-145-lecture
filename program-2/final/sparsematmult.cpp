#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include<algorithm>
#include <bits/stdc++.h>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

void mysort(csr_t *mat, ptr_t start, ptr_t end) {
  vector<pair<ptr_t, val_t>> row;

  // put ind/val pairs from row into vector of pairs
  for(ptr_t i = start; i < end; i++) {
    row.push_back(make_pair(mat->ind[i], mat->val[i]));
  }

  // sort vector by ind
  sort(row.begin(), row.end());

  // put ind/val pairs back into csr 
  for(ptr_t i = start; i < end; i++) {
    mat->ind[i] = row[i-start].first;
    mat->val[i] = row[i-start].second;
  }
}

void reorder(csr_t * mat) {
  for(idx_t i = 0; i < mat->nrows; i++) {
    mysort(mat, mat->ptr[i], mat->ptr[i+1]);
  }
}

/**
 * Multiply A and B (transposed given) and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void sparsematmult(csr_t * A, csr_t * B, csr_t *C, ptr_t c_nnz, double factor)
{
  ptr_t cnnz_max = (ptr_t) (A->nrows * B->ncols * factor);
  C->reserve(A->nrows, cnnz_max);
  C->ncols = B->nrows;

  C->ptr[0] = 0;
  
  ptr_t cnnz = 0;
  idx_t i = 0;
  idx_t j = 0;
  idx_t l = 0;
  idx_t k = 0;
  ptr_t nnz = 0;
  
  for(i = 0; i < A->nrows; i++) {
    if(cnnz > cnnz_max)
        C->reserve(C->nrows, cnnz*(1+(val_t)(i/C->nrows)));
    nnz = 0;
    #pragma omp parallel for schedule(static) shared(A,B,C,nnz) firstprivate(i)
    for(j = 0; j < B->nrows; j++) { 
      k = A->ptr[i];
      l = B->ptr[j];
      val_t sum = 0;
      while(k < A->ptr[i+1] && l < B->ptr[j+1]) {
        if(A->ind[k] == B->ind[l]) {
          //cout << "Match: " << A->ind[k] << " " << B->ind[l] << endl;
          //cout << "(" << A->val[k] << " * " << B->val[l] << ") + " << sum << " = ";
          sum += A->val[k] * B->val[l];
          //cout << sum << endl;
          k++;
          l++;
        } else if(A->ind[k] < B->ind[l]) {
          //cout << "A Less: " <<A->ind[k] << " " << B->ind[l] << endl;
          k++;
        } else {
          //cout << "B Less: " <<A->ind[k] << " " << B->ind[l] << endl;
          l++;
        }
      }
      //cout << "ended row B" << endl;

      if(sum != 0.0) {
        //cout << "Final Sum: " << sum << endl;
        cnnz++;
        C->ind[C->ptr[i] + nnz] = j;
        C->val[C->ptr[i] + nnz] = sum;
        nnz++;
      }
    }
    C->ptr[i+1] = cnnz;
  }
}



void sparsematmult_omp(csr_t * A, csr_t * B, csr_t *C, double factor)
{
  ptr_t cnnz_max = (ptr_t) (A->nrows * B->nrows);
  C->reserve(A->nrows, cnnz_max);
  C->ncols = B->nrows;
  

  C->ptr[0] = 0;
  
  ptr_t cnnz = 0;
  idx_t i = 0;
  idx_t j = 0;
  idx_t l = 0;
  idx_t k = 0;
  ptr_t nnz = 0;
  
  for(i = 0; i < A->nrows; i++) {
    nnz = 0;
    #pragma omp parallel for schedule(static) shared(A,B,C,cnnz,cnnz_max,nnz) firstprivate(i) private(j,k,l)
    for(j = 0; j < B->nrows; j++) { 
      k = A->ptr[i];
      l = B->ptr[j];
      val_t sum = 0;
      while(k < A->ptr[i+1] && l < B->ptr[j+1]) {
        if(A->ind[k] == B->ind[l]) {
          sum += A->val[k] * B->val[l];
          k++;
          l++;
        } else if(A->ind[k] < B->ind[l]) {
          k++;
        } else {
          l++;
        }
      }

      if(sum != 0.0) {
        if(cnnz >= cnnz_max) {
          #pragma omp critical
          {
            C->reserve(A->nrows, cnnz_max * cnnz_max);
          }
        }
        #pragma omp critical 
        {
          cnnz++;
          C->ind[C->ptr[i] + nnz] = j;
          C->val[C->ptr[i] + nnz] = sum;
          nnz++;
        }
      }
    }
    C->ptr[i+1] = cnnz;
  }
}

void print_attributes(csr_t *mat, string name) {
  cout << name << "->ptr: [";
  for(idx_t i = 0; i <= mat->nrows; i++) {
      cout << mat->ptr[i] << ",";
  }
  cout << "]" << endl;
  cout << name << "->ind[";
  for(ptr_t i = 0; i < mat->ptr[mat->nrows]; i++) {
      cout << mat->ind[i] << ",";
  }
  cout << "]" << endl;
  cout << name << "->val[";
  for(ptr_t i = 0; i < mat->ptr[mat->nrows]; i++) {
      cout << mat->val[i] << ",";
  }
  cout << "]" << endl;
}

void to_matrix(csr_t *mat, vector<vector<float>> &aa) {
  for(idx_t i = 0; i < mat->nrows; i++) {
    for(ptr_t j = 0; j < mat->ptr[i+1]-mat->ptr[i]; j++) {
        aa[i][mat->ind[j+mat->ptr[i]]] = mat->val[j+mat->ptr[i]];
    }
  }
}

void to_file(vector<vector<float>> mat, ofstream &myfile) {
  for(idx_t i = 0; i < mat.size(); i++) {
    for(idx_t j = 0; j < mat[0].size(); j++) {
      myfile << mat[i][j] << " ";
    }
    myfile << endl;
  }
}

void solve_trivial(vector<vector<float>> &real, vector<vector<float>> mat1, vector<vector<float>> mat2) {
  for(long unsigned int i = 0; i < mat1.size(); i++) {
    for(long unsigned int j = 0; j < mat2[0].size(); j++) {
      for(long unsigned int k = 0; k < mat1[0].size(); k++) {
        real[i][j] += mat1[i][k] * mat2[j][k];
      }
    }
  }
}

int main(int argc, char *argv[])
{
  /* init args */
  if(argc < 4){
    cerr << "Invalid options." << endl << "<program> <A_nrows> <A_ncols> <B_ncols> <fill_factor> [-t <num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);
  int nthreads = 1;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }
  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols2, ncols, factor); // Note B is already transposed.
  test_matrix(A);
  test_matrix(B);

  reorder(A);
  reorder(B);
  
  auto C = new csr_t(); // Note that C has no data allocations so far.

  cout << A->info("A") << endl;
  cout << B->info("B") << endl;

  auto t1 = omp_get_wtime();
  sparsematmult_omp(A, B, C, factor);
  auto t2 = omp_get_wtime();

  cout << C->info("C") << endl;

  // print_attributes(A, "A");
  // print_attributes(B, "B");
  // print_attributes(C, "C");

  // std::ofstream a_file("a.txt");
  // std::ofstream b_file("b.txt");
  // std::ofstream c_file("c.txt");
  // std::ofstream actual_file("actual.txt");

  // vector<vector<float>> aa(A->nrows, vector<float> (A->ncols, 0.0));
  // vector<vector<float>> bb(B->nrows, vector<float> (B->ncols, 0.0));
  // vector<vector<float>> cc(C->nrows, vector<float> (C->ncols, 0.0));
  // vector<vector<float>> actual(nrows, vector<float> (ncols2, 0.0));

  // to_matrix(A, aa);
  // to_matrix(B, bb);
  // to_matrix(C, cc);
  // solve_trivial(actual, aa, bb);

  // to_file(aa, a_file);
  // to_file(bb, b_file);
  // to_file(cc, c_file);
  // to_file(actual, actual_file);

  // a_file.close();
  // b_file.close();
  // c_file.close();
  // actual_file.close();

  cout << "Execution time: " << (t2-t1) << endl;

  std::ofstream output;
  string name = argv[4];
  name += ".csv";
	output.open(name, std::ofstream::app);

  output << endl << nthreads << " " << t2-t1;

  delete A;
  delete B;
  delete C;

  return 0;
}
