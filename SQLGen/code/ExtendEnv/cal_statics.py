import pandas as pd
import os


type_map = {
    'select': 1,
    'update': 2,
    'insert': 3,
    'delete': 4,
}


def cal_statics(path):
    if not os.path.exists(path):
        return
    s_path = path + '_statics'

    with open(path, 'r') as f:
        sqls = f.read().split('\n')

        s_data = pd.DataFrame(columns=['q_type', 'join', 'aggregate', 'nest', 'having', 'predicate',
                                       'length', 'order by'])
        print(len(sqls))
        for sql in sqls:
            if sql == "":
                continue
            sql = sql.replace("=", " = ")
            sql = sql.replace("  ", " ")
            words = sql.split(' ')
            q_type = type_map[words[0]]
            join_nums = 0
            aggregate = 0
            nest = 0
            having = 0
            predicate = 0
            length = len(words)
            order_by = 0
            for word in words:
                if word == 'join':
                    join_nums += 1
                if word == 'where' or word == 'and':
                    predicate += 1
                if word == 'order':
                    order_by = 1
                if word == 'having':
                    having = 1
                if word == '(select':
                    nest = 1
                if word == 'group':
                    aggregate = 1

            row = {'q_type': q_type, 'join': join_nums, 'aggregate': aggregate, 'nest': nest, 'having': having,
                   'predicate': predicate, 'length': length, 'order by': order_by}

            s_data.loc[s_data.shape[0]] = row
            s_data.to_csv(s_path)


def show_dict(a_dict, add_label=""):
    s_dict = sorted(a_dict)
    list_name = []
    list_value = []
    for key in s_dict:
        if add_label != "":
            list_name.append("{} {}".format(key, add_label))
        else:
            list_name.append(key)
        list_value.append(a_dict[key])
    return list_name, list_value


def cal_proportion(path):
    if not os.path.exists(path):
        return
    statics = pd.read_csv(path, index_col=0)
    total_nums = statics.shape[0]
    type_dict = dict(statics['q_type'].value_counts())
    # select_query = statics.where(statics['q_type'] == type_map['select'])
    select_nums = type_dict[type_map['select']]
    other_nums = total_nums - select_nums
    join_dict = dict(statics['join'].value_counts())
    join_dict[0] -= other_nums
    aggregate_dict = dict(statics['aggregate'].value_counts())
    aggregate_dict[0] -= other_nums
    nest_dict = dict(statics['nest'].value_counts())
    nest_dict[0] -= other_nums
    having_dict = dict(statics['having'].value_counts())
    having_dict[0] -= other_nums
    predicate_dict = dict(statics['predicate'].value_counts())
    length_dict = dict(statics['length'].value_counts())
    order_by_dict = dict(statics['order by'].value_counts())
    order_by_dict[0] -= other_nums
    print('type:', type_dict)
    print('join:', join_dict)
    print('aggregate:',  aggregate_dict)
    print('nest:', nest_dict)
    print('having:', having_dict)
    print('predicate:', predicate_dict)
    print('length:', length_dict)
    print('order by:', order_by_dict)
    pred_name, pred_value = show_dict(predicate_dict, "pred.")
    print('sort predicate:', pred_name, pred_value)
    nest_name, nest_value = show_dict(nest_dict)
    print('sort nest:', nest_name, nest_value)
    len_name, len_value = show_dict(length_dict)
    print('len sort:', len_name, len_value)


if __name__ == '__main__':
    path = os.path.abspath('.') + '/tpch/cost_pc1000000_sql'
    cal_statics(path)
    path = os.path.abspath('.') + '/tpch/cost_pc1000000_sql_statics'
    cal_proportion(path)