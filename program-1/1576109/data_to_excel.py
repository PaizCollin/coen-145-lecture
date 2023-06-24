# import modules
from this import s
import pandas as pd

# main function
if __name__ == '__main__':
    serial = pd.read_csv('serial.csv', sep=' ')
    block = pd.read_csv('block.csv', sep=' ')
    tiling = pd.read_csv('tiling.csv', sep=' ')

    with pd.ExcelWriter('results.xlsx') as writer:
        serial.to_excel(writer, sheet_name='Serial')
        block.to_excel(writer, sheet_name='Block')
        tiling.to_excel(writer, sheet_name='Tiling')

# end