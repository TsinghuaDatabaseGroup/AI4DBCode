import re
import pandas as pd

class TreeNode(object):
    def __init__(self, current_vec, parent):
        self.item = current_vec
        self.parent = parent
        self.children = []

    def get_parent(self):
        return self.parent

    def get_item(self):
        return self.item

    def get_children(self):
        return self.children

    def add_child(self, child):
        self.children.append(child)


def recover_tree(vecs, parent):
    if len(vecs) == 0:
        return vecs
    if vecs[0] is None:
        return vecs[1:]
    node = TreeNode(vecs[0], parent)
    while True:
        vecs = recover_tree(vecs[1:], node)
        parent.add_child(node)
        if len(vecs) == 0:
            return vecs
        if vecs[0] is None:
            return vecs[1:]
        node = TreeNode(vecs[0], parent)


def bitand(bit1, bit2):
    if len(bit1) > 0 and len(bit2) > 0:
        result = []
        for i in range(len(bit1)):
            result.append(min(bit1[i], bit2[i]))
        return result
    elif len(bit1) > 0:
        return bit1
    elif len(bit2) > 0:
        return bit2
    else:
        return []


def bitor(bit1, bit2):
    if len(bit1) > 0 and len(bit2) > 0:
        result = []
        for i in range(len(bit1)):
            result.append(max(bit1[i], bit2[i]))
        return result
    elif len(bit1) > 0:
        return bit1
    elif len(bit2) > 0:
        return bit2
    else:
        return []


def get_bitmap(root, data, sample, sample_num):
    predicate = root.get_item()
    if predicate is not None and predicate['op_type'] == 'Compare':
        table_name = predicate['left_value'].split('.')[0]
        column = predicate['left_value'].split('.')[1]
        vec = []
        pattern = re.compile(r'^[a-z_]+\.[a-z][a-z0-9_]*$')
        result = pattern.match(predicate['right_value'])
        if result is None:
            dtype = data[table_name].dtypes[column]
            for value in sample[table_name][column].tolist():
                if isSelected(value, predicate, dtype):
                    vec.append(1)
                else:
                    vec.append(0)
            for i in range(len(vec), sample_num):
                vec.append(0)
        elif not predicate['right_value'].split('.')[0] in data:
            dtype = data[table_name].dtypes[column]
            for value in sample[table_name][column].tolist():
                if isSelected(value, predicate, dtype):
                    vec.append(1)
                else:
                    vec.append(0)
            for i in range(len(vec), sample_num):
                vec.append(0)
        return vec
    elif predicate is not None and predicate['op_type'] == 'Bool':
        bitmap = []
        if predicate['operator'] == 'AND':
            for child in root.get_children():
                vec = get_bitmap(child, data, sample, sample_num)
                bitmap = bitand(bitmap, vec)
        elif predicate['operator'] == 'OR':
            for child in root.get_children():
                vec = get_bitmap(child, data, sample, sample_num)
                bitmap = bitor(bitmap, vec)
        else:
            print(predicate['operator'])
            raise
        return bitmap
    else:
        return []


def prepare_samples(data, sample_num, tables):
    sample = dict()
    for table_name in tables:
        sample[table_name] = data[table_name].sample(n=min(sample_num, len(data[table_name])))
    return sample


# Second, we should encode each training instance one by one except for the label
def isSelected(row_value, predicate, dtype):
    if dtype == 'int64':
        row_value = int(row_value)
        value = int(predicate['right_value'])
        op = predicate['operator']
        if op == '=':
            if row_value != value:
                return False
        elif op == '!=':
            if row_value == value:
                return False
        elif op == '<':
            if row_value >= value:
                return False
        elif op == '>':
            if row_value <= value:
                return False
        elif op == '<=':
            if row_value > value:
                return False
        elif op == '>=':
            if row_value < value:
                return False
        else:
            print(op)
            raise
    elif dtype == 'float64':
        row_value = float(row_value)
        value = float(predicate['right_value'])
        op = predicate['operator']
        if op == '=':
            if row_value != value:
                return False
        elif op == '!=':
            if row_value == value:
                return False
        elif op == '<':
            if row_value >= value:
                return False
        elif op == '>':
            if row_value <= value:
                return False
        elif op == '<=':
            if row_value > value:
                return False
        elif op == '>=':
            if row_value < value:
                return False
        else:
            print(op)
            raise
    elif dtype == 'object':
        value = predicate['right_value']
        op = predicate['operator']
        if pd.isnull(row_value):
            row_value = ''
        else:
            row_value = str(row_value)
        if op == '=':
            if value.startswith('__LIKE__'):
                v = value[8:]
                pattern = r'^'
                for idx, token in enumerate(v.split('%')):
                    if len(token) == 0:
                        pattern += r'.*'
                    else:
                        pattern += re.escape(token)
                        if idx < len(v.split('%')) - 1:
                            pattern += r'.*'
                pattern += r'$'
                if re.match(pattern, row_value) == None:
                    return False
            elif value.startswith('__NOTLIKE__'):
                v = value[11:]
                pattern = r'^'
                for idx, token in enumerate(v.split('%')):
                    if len(token) == 0:
                        pattern += r'.*'
                    else:
                        pattern += re.escape(token)
                        if idx < len(v.split('%')) - 1:
                            pattern += r'.*'
                pattern += r'$'
                if re.match(pattern, row_value) != None:
                    return False
            elif value.startswith('__NOTEQUAL__'):
                pattern = value[12:]
                if row_value == pattern:
                    return False
            elif value.startswith('__ANY__'):
                pattern = value.strip('__ANY__')
                pattern = pattern.strip('{}')
                for token in pattern.split(','):
                    token = token.strip('"').strip('\'')
                    if row_value == token:
                        return True
                return False
            elif value == 'None':
                if len(row_value) > 0:
                    return False
            else:
                if row_value != value:
                    return False
        elif op == 'IS':
            if value == 'None':
                if len(row_value) > 0:
                    return False
            else:
                print(value)
                raise
        elif op == '!=':
            if value == 'None':
                if len(row_value) == 0:
                    return False
            else:
                if row_value == value:
                    return False
        elif op == '<':
            if row_value >= value:
                return False
        elif op == '>':
            if row_value <= value:
                return False
        elif op == '<=':
            if row_value > value:
                return False
        elif op == '>=':
            if row_value < value:
                return False
        else:
            print(op)
            raise
    else:
        print(dtype)
        raise
    return True
