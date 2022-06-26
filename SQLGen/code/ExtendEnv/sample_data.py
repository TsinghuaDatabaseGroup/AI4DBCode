import os
import base


class MetaDataSupport:
    def __init__(self, dbname, target_cardinality):
        self.db, self.cursor = base.connect_server(dbname)
        self.dbname = dbname + '_4'
        self.schema = base.get_table_structure(dbname)
        # print(self.schema)
        self.task = target_cardinality
        self.statistics = self.load_statistics()
        self.numeric_column = ['s_acctbal', 'c_acctbal', 'p_retailprice', 'ps_supplycost', 'o_totalprice', 'l_extendedprice']
        print('MetaDataSupport init')

    def cal_nums(self, table_name, field):
        return 100
        # threshold = self.task * 0.1
        # if self.statistics[table_name][field]['count'] < threshold:
        #     return min(self.statistics[table_name][field]['count'], 1000)
        # else:
        #     nums = self.statistics[table_name][field]['count'] / threshold
        # return nums

    def load_statistics(self, refresh=False):
        print("loading statistics...")
        stat_path = os.path.abspath('.') + '/sampled_data/' + self.dbname + '/statistics'
        if os.path.exists(stat_path) and refresh is False:
            with open(stat_path, 'r') as f:
                statistics = eval(f.read())
            f.close()
            print("statistics load done...")
            return statistics
        print("statistics refreshing...")
        statistics = {}
        for table_name in self.schema:
            statistics[table_name] = {}
            for field in self.schema[table_name]:
                statistics[table_name][field] = {}
                print(field, '-', table_name)
                sql = "select count({0}), max({0}), min({0}), count(distinct {0}) from {1}".format(field, table_name)
                self.cursor.execute(sql)
                data = self.cursor.fetchall()
                print(data)
                statistics[table_name][field]['count'] = data[0][0]
                statistics[table_name][field]['distinct'] = data[0][3]
                if type(data[0][1]) is not int:
                    statistics[table_name][field]['max'] = float(data[0][1])
                else:
                    statistics[table_name][field]['max'] = data[0][1]
                if type(data[0][2]) is not int:
                    statistics[table_name][field]['min'] = float(data[0][2])
                else:
                    statistics[table_name][field]['min'] = data[0][2]

                sql = "select tablename, attname, n_distinct, most_common_vals, histogram_bounds " \
                      "from pg_stats " \
                      "where tablename = '{}' and attname = '{}'".format(table_name, field)
                self.cursor.execute(sql)
                data = self.cursor.fetchall()

                if len(data) != 0:
                    if data[0][3] is not None:
                        statistics[table_name][field]['most_common_vals'] = eval(data[0][3])
                    if data[0][4] is not None:
                        print("table:{}-field:{}-histogram_bounds:{}".format(table_name, field, len(data[0][4])))
                        statistics[table_name][field]['histogram_bounds'] = eval(data[0][4])

        with open(stat_path, 'w') as f:
            f.write(str(statistics))
        f.close()
        print("load statistics done...")
        return statistics

    def sample_field_data2(self, table_name, field):
        if field in self.numeric_column:
            nums = min(int(self.statistics[table_name][field]['distinct'] * 0.01), 400)
        else:
            nums = 100
        if self.statistics[table_name][field]['distinct'] <= nums:
            field_data = list()
            sql = "select distinct {0} from {1} where {0} is not null".format(field, table_name)
            self.cursor.execute(sql)
            data = self.cursor.fetchall()
            for col in data:
                field_data.append(int(col[0]))
            field_data.sort()
            print("sample {} {} done. nums: {}. from: distinct".format(table_name, field, len(field_data)))
            return tuple(field_data)

        step = int((self.statistics[table_name][field]['max'] - self.statistics[table_name][field]['min']) / nums)
        field_data = list()
        field_data.append(int(self.statistics[table_name][field]['min']))
        for index in range(nums-1):
            field_data.append(field_data[index] + step)
        print("sample {} {} done. nums: {}. from: random".format(table_name, field, len(field_data)))
        return tuple(field_data)

    def load_static_data(self, path):
        """
        文件读取
        :param path:
        :return: sample_data {table_name {field: set()}}
        """
        sample_data = {}
        for table_name in self.schema:
            sample_data[table_name] = {}
            table_path = path + '/' + table_name
            if not os.path.exists(table_path):
                os.mkdir(table_path)
            for field in self.schema[table_name]:
                field_path = table_path + '/' + field
                if not os.path.exists(field_path):
                    sample_data[table_name][field] = self.sample_field_data2(table_name, field)
                    with open(field_path, 'w') as f:
                        f.write(str(sample_data[table_name][field]))
                    f.close()
                else:
                    with open(field_path, 'r') as f:
                        sample_data[table_name][field] = eval(f.read())
                        print("{0}-{1}:{2}".format(table_name, field, len(sample_data[table_name][field])))
                    f.close()
        return sample_data

    def get_data(self, refresh=False):
        sample_path = os.path.abspath('.') + '/sampled_data/' + self.dbname
        if os.path.exists(sample_path) and refresh is False:
            print('load static sample data')
            return self.load_static_data(sample_path)
        print("refresh sample data...")
        if not os.path.exists(sample_path):
            os.mkdir(sample_path)
        sample_data = {}
        for table_name in self.schema:
            sample_data[table_name] = {}
            table_path = sample_path + '/' + table_name
            if not os.path.exists(table_path):
                os.mkdir(table_path)
            for field in self.schema[table_name]:
                sample_data[table_name][field] = self.sample_field_data2(table_name, field)
                field_path = table_path + '/' + field
                with open(field_path, 'w') as f:
                    f.write(str(sample_data[table_name][field]))
                f.close()
        print("sample data done.")
        return sample_data


if __name__ == '__main__':
    a = MetaDataSupport('tpch', 1000)
    # a.load_statistics(refresh=True)
    a.get_data(refresh=True)
    # a.sample_field_data2('orders', 'o_orderkey')
    # print(a.statistics)
    # a.get_data(refresh=True)

    # a.get_data(refresh=True)



