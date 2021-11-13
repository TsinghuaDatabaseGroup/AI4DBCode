"""Dataset registrations."""
import common
import numpy as np
import pandas as pd

'''
def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record Type', 'Registration Class', 'State', 'County', 'Body Type',
        'Fuel Type', 'Reg Valid Date', 'Color', 'Scofflaw Indicator',
        'Suspension Indicator', 'Revocation Indicator'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    return common.CsvTable('DMV', csv_file, cols, type_casts)
'''


def LoadDmv(filename='low.csv'):  # modify
    csv_file = '../train-test-data/forest_power-data-sql/{}'.format(filename)
    table_head = pd.read_csv(csv_file, sep=',', escapechar='\\', encoding='utf-8', low_memory=False, quotechar='"')
    cols = list(table_head.columns)
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg Valid Date': np.datetime64}
    # ptint('csvfile',csv_file)
    return common.CsvTable('DMV', csv_file, cols, type_casts)
