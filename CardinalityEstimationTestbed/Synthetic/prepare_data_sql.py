import os

# 
for cols in [2, 4, 6, 8]:
    for distinct in [10, 100, 1000, 10000]:
        for corr in [2, 4, 6, 8]:
            for skew in [2, 4, 6, 8]:
                generate_data_sql = 'python generate_data_sql.py --cols ' + str(cols) + ' --distinct ' + str(
                    distinct) + ' --corr ' + str(corr) + ' --skew ' + str(skew)
                get_truecard = 'python get_truecard.py --version cols_' + str(cols) + '_distinct_' + str(
                    distinct) + '_corr_' + str(corr) + '_skew_' + str(skew)

                os.system(generate_data_sql)
                os.system(get_truecard)
                print('cols_' + str(cols) + '_distinct_' + str(distinct) + '_corr_' + str(corr) + '_skew_' + str(
                    skew) + 'is prepared.')
