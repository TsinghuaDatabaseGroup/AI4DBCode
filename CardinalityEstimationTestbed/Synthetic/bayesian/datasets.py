"""Dataset registrations."""

import common
import numpy as np


def LoadDmv(filename='Vehicle__Snowmobile__and_Boat_Registrations.csv'):
    csv_file = './datasets/{}'.format(filename)
    cols = [
        'Record_Type', 'Registration_Class', 'State', 'County', 'Body_Type',
        'Fuel_Type', 'Reg_Valid_Date', 'Color', 'Scofflaw_Indicator',
        'Suspension_Indicator', 'Revocation_Indicator', 'Zip', 'Unladen_Weight'
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {'Reg_Valid_Date': np.datetime64, 'Zip': np.int32, 'Unladen_Weight': np.int32}
    return common.CsvTable('DMV', csv_file, cols, type_casts)


def LoadStoreSales(filename='store_sales.csv'):
    csv_file = '/home/jintao/naru/csvdata_sql/{}'.format(filename)
    cols = [
        "ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk", "ss_cdemo_sk",
        "ss_hdemo_sk", "ss_wholesale_cost", "ss_list_price", "ss_sales_price", "ss_ext_discount_amt",
        "ss_ext_sales_price"
    ]
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {"ss_sold_date_sk": np.int32, "ss_sold_time_sk": np.int32, "ss_item_sk": np.int32,
                  "ss_customer_sk": np.int32, "ss_cdemo_sk": np.int32,
                  "ss_hdemo_sk": np.int32, "ss_wholesale_cost": np.float32, "ss_list_price": np.float32,
                  "ss_sales_price": np.float32, "ss_ext_discount_amt": np.float32,
                  "ss_ext_sales_price": np.float32}
    return common.CsvTable('store_sales', csv_file, cols, type_casts)


def LoadCDCS(paras):
    csv_name = 'cols_{}_distinct_{}_corr_{}_skew_{}'.format(paras[0], paras[1], paras[2], paras[3])
    csv_file = '../csvdata_sql/{}.csv'.format(csv_name)
    cols = []
    for i in range(paras[0]):
        cols.append("col{}".format(i))
    # Note: other columns are converted to objects/strings automatically.  We
    # don't need to specify a type-cast for those because the desired order
    # there is the same as the default str-ordering (lexicographical).
    type_casts = {}
    for col in cols:
        type_casts[col] = np.int32
    return common.CsvTable(csv_name, csv_file, cols, type_casts)
