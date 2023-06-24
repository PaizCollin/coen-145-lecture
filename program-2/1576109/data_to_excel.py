# import modules
from this import s
import pandas as pd

# main function
if __name__ == '__main__':
    fill1 = pd.read_csv('0.05.csv', sep=' ')
    fill2 = pd.read_csv('0.1.csv', sep=' ')
    fill3 = pd.read_csv('0.15.csv', sep=' ')
    fill4 = pd.read_csv('0.2.csv', sep=' ')

    with pd.ExcelWriter('results.xlsx') as writer:
        fill1.to_excel(writer, sheet_name='0.05')
        fill2.to_excel(writer, sheet_name='0.1')
        fill3.to_excel(writer, sheet_name='0.15')
        fill4.to_excel(writer, sheet_name='0.2')

# end