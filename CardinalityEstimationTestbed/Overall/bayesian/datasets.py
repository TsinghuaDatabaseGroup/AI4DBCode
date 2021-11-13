"""Dataset registrations."""

import common
import numpy as np
import pandas as pd


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


def LoadJobTables(filename=''):
    csv_file = filename
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {}
    dataset = pd.read_csv(csv_file)
    all_cols = {
        'title': [
            'kind_id', 'production_year', 'episode_nr', 'imdb_index', 'phonetic_code', 'season_nr', 'series_years'
        ],
        'cast_info': [
            'nr_order', 'role_id'
        ],
        'movie_companies': [
            'company_type_id'
        ],
        'movie_info_idx': ['info_type_id'],
        'movie_info': ['info_type_id'],
        'movie_keyword': ['keyword_id']
    }
    table_name = filename.split('/')[0].split('.')[0]
    if table_name in all_cols:
        cols = all_cols[table_name]
    else:
        cols = dataset.columns
        dataset = dataset[cols]
    return common.CsvTable('JOB', dataset, dataset.columns, type_casts)
