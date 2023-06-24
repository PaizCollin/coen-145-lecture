# import modules
from this import s
import pandas as pd

# main function
if __name__ == '__main__':
    fill1 = pd.read_csv(r'C:\Users\User\OneDrive - scu.edu\School\4\COEN 145\HW2\Results\Test3\05.csv', sep=' ')
    fill2 = pd.read_csv(r'C:\Users\User\OneDrive - scu.edu\School\4\COEN 145\HW2\Results\Test3\1.csv', sep=' ')
    fill3 = pd.read_csv(r'C:\Users\User\OneDrive - scu.edu\School\4\COEN 145\HW2\Results\Test3\15.csv', sep=' ')
    fill4 = pd.read_csv(r'C:\Users\User\OneDrive - scu.edu\School\4\COEN 145\HW2\Results\Test3\2.csv', sep=' ')

    with pd.ExcelWriter(r'C:\Users\User\OneDrive - scu.edu\School\4\COEN 145\HW2\Results\Test3\results3.xlsx') as writer:
        fill1.to_excel(writer, sheet_name='3_0.05')
        fill2.to_excel(writer, sheet_name='3_0.1')
        fill3.to_excel(writer, sheet_name='3_0.15')
        fill4.to_excel(writer, sheet_name='3_0.2')

# end