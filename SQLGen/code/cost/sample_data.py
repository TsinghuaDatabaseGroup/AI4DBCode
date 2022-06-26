import os
import base
import datetime


class MetaDataSupport:
    def __init__(self, dbname):
        self.db, self.cursor = base.connect_server(dbname)
        self.dbname = dbname
        self.schema = base.get_table_structure(dbname)
        # print(self.schema)
        # self.task = target_cardinality
        self.statistics = self.load_statistics()

    # def cal_nums(self, table_name, field):
    #     return 100
    #     threshold = self.task * 0.1
    #     if self.statistics[table_name][field]['count'] < threshold:
    #         return min(self.statistics[table_name][field]['count'], 1000)
    #     else:
    #         nums = self.statistics[table_name][field]['count'] / threshold
    #     return nums

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
                # if type(data[0][1]) is str:
                #     statistics[table_name][field]['max'] = data[0][1]
                # else:
                #     if type(data[0][1]) is not int or type(data[0][1]) is not datetime.date:
                #         statistics[table_name][field]['max'] = float(data[0][1])
                #     else:
                #         statistics[table_name][field]['max'] = data[0][1]
                #
                # if type(data[0][2]) is str:
                #     statistics[table_name][field]['max'] = data[0][2]
                # else:
                #     if type(data[0][2]) is not int or type(data[0][1]) is not datetime.date:
                #         statistics[table_name][field]['min'] = float(data[0][2])
                #     else:
                #         statistics[table_name][field]['min'] = data[0][2]
                sql = "select tablename, attname, n_distinct, most_common_vals, histogram_bounds " \
                      "from pg_stats " \
                      "where tablename = '{}' and attname = '{}'".format(table_name, field)
                self.cursor.execute(sql)
                data = self.cursor.fetchall()

                if len(data) != 0:
                    if data[0][3] is not None:
                        try:
                            value = eval(data[0][3])
                        except Exception as result:
                            value = tuple(data[0][3][1:-1].split(','))
                        statistics[table_name][field]['most_common_vals'] = value
                        print("table:{}-field:{}-most_common_vals:{}".format(table_name, field, len(value)))
                    if data[0][4] is not None:
                        try:
                            value = eval(data[0][4])
                        except Exception as result:
                            value = tuple(data[0][4][1:-1].split(','))
                        statistics[table_name][field]['histogram_bounds'] = value
                        print("table:{}-field:{}-histogram_bounds:{}".format(table_name, field, len(value)))

        with open(stat_path, 'w') as f:
            f.write(str(statistics))
        f.close()
        print("load statistics done...")
        return statistics

    def sample_field_data2(self, table_name, field):
        if 'histogram_bounds' in self.statistics[table_name][field].keys():
            data = self.statistics[table_name][field]['histogram_bounds']
            # print(type(data))
            print("sample {} {} done. nums: {}. from: histogram_bounds".format(table_name, field,
                                                       len(data)))
            return tuple(data)
        if 'most_common_vals' in self.statistics[table_name][field].keys():
            print("sample {} {} done. nums: {}. from: most_common_vals".format(table_name, field,
                                                       len(self.statistics[table_name][field]['most_common_vals'])))
            return tuple(self.statistics[table_name][field]['most_common_vals'])
        nums = 100
        if self.statistics[table_name][field]['distinct'] <= nums:
            field_data = list()
            sql = "select distinct {} from {}".format(field, table_name)
            self.cursor.execute(sql)
            data = self.cursor.fetchall()
            for col in data:
                if col[0] is None:
                    continue
                if type(col[0]) is str:
                    field_data.append(col[0])
                else:
                    if type(col[0]) is not int:
                        field_data.append(float(col[0]))
                    else:
                        field_data.append(col[0])
            field_data.sort()
            print("sample {} {} done. nums: {}. from: distinct".format(table_name, field, len(field_data)))
            return tuple(field_data)

        field_data = set()
        if len(field_data) < nums:
            sql = "select {} from {} order by random() limit {}".format(field, table_name, nums - len(field_data))
            self.cursor.execute(sql)
            data = self.cursor.fetchall()
            field_data = set()
            for col in data:
                if type(col[0]) is str:
                    field_data.add(col[0])
                else:
                    if type(col[0]) is not int:
                        field_data.add(float(col[0]))
                    else:
                        field_data.add(col[0])
        field_data = list(field_data)
        field_data.sort()
        print("sample {} {} done. nums: {}. from: ".format(table_name, field, len(field_data)))
        return tuple(field_data)

    # def sample_field_data(self, table_name, field):
    #     """
    #     返回的是tuple类型且无重复的采样数据
    #     :param table_name:
    #     :param field:
    #     :param nums:
    #     :return: tuple
    #     """
    #     nums = self.cal_nums(table_name, field)
    #     # print("sample {} {} done. nums: {}".format(table_name, field, nums))
    #     if self.statistics[table_name][field]['distinct'] <= nums:
    #         field_data = list()
    #         sql = "select distinct {} from {}".format(field, table_name)
    #         self.cursor.execute(sql)
    #         data = self.cursor.fetchall()
    #         for col in data:
    #             if type(col[0]) is not int:
    #                 field_data.append(float(col[0]))
    #             else:
    #                 field_data.append(col[0])
    #         field_data.sort()
    #         print("sample {} {} done. nums: {}".format(table_name, field, len(field_data)))
    #         return tuple(field_data)
    #     else:
    #         field_data = set()
    #         field_data.add(self.statistics[table_name][field]['max'])
    #         field_data.add(self.statistics[table_name][field]['min'])
    #         # if 'most_common_vals' in self.statistics[table_name][field].keys():
    #         #     field_data.add(self.statistics[table_name][field]['most_common_vals'])
    #         if len(field_data) < nums and 'histogram_bounds' in self.statistics[table_name][field].keys():
    #             field_data.add(self.statistics[table_name][field]['histogram_bounds'])
    #         if len(field_data) < nums:
    #             sql = "select {} from {} order by random() limit {}".format(field, table_name, nums-len(field_data))
    #             self.cursor.execute(sql)
    #             data = self.cursor.fetchall()
    #             field_data = set()
    #             for col in data:
    #                 if type(col[0]) is not int:
    #                     field_data.add(float(col[0]))
    #                 else:
    #                     field_data.add(col[0])
    #     field_data = list(field_data)
    #     field_data.sort()
    #     print("sample {} {} done. nums: {}".format(table_name, field, len(field_data)))
    #     return tuple(field_data)

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
                        print("{0}-{1}".format(table_name, field))
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
    a = MetaDataSupport('tpch')
    # a.load_statistics(refresh=True)
    a.get_data()
    # a.sample_field_data2('orders', 'o_orderkey')
    # print(a.statistics)
    # a.get_data(refresh=True)

    # a.get_data(refresh=True)



