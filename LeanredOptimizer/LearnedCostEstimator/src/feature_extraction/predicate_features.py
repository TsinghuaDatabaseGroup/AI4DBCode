import re

import pypred

from src.feature_extraction.predicate_operators import *


def remove_invalid_tokens(predicate):
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text ~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__LIKE__\2')", predicate)
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text !~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTLIKE__\2')", x)
    x = re.sub(r'\(\(([a-zA-Z_]+)\)::text <> \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTEQUAL__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) ~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__LIKE__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) !~~ \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTLIKE__\2')", x)
    x = re.sub(r'\(([a-zA-Z_]+) <> \'(((?!::text).)*)\'::text\)', r"(\1 = '__NOTEQUAL__\2')", x)
    x = re.sub(r'(\'[^\']*\')::[a-z_]+', r'\1', x)
    x = re.sub(r'\(([^\(]+)\)::[a-z_]+', r'\1', x)
    x = re.sub(r'\(([a-z_0-9A-Z\-]+) = ANY \(\'(\{.+\})\'\[\]\)\)', r"(\1 = '__ANY__\2')", x)
    return x


def predicates2seq(pre_tree, alias2table, relation_name, index_name):
    current_level = -1
    current_line = 0
    sequence = []
    while current_line < len(pre_tree):
        operator_str = pre_tree[current_line]
        level = len(re.findall(r'\t', operator_str))
        operator_seq = operator_str.strip('\t').split(' ')
        operator_type = operator_seq[1]
        operator = operator_seq[0]
        if level <= current_level:
            for i in range(current_level - level + 1):
                sequence.append(None)
        current_level = level
        if operator_type == 'operator':
            sequence.append(Operator(operator))
            current_line += 1
        elif operator_type == 'comparison':
            operator = operator_seq[0]
            current_line += 1
            operator_str = pre_tree[current_line]
            operator_seq = operator_str.strip('\t').split(' ')
            left_type = operator_seq[0]
            left_value = operator_seq[1]
            current_line += 1
            operator_str = pre_tree[current_line]
            operator_seq = operator_str.strip('\t').split(' ')
            right_type = operator_seq[0]
            if right_type == 'Number':
                right_value = operator_seq[1]
            elif right_type == 'Literal':
                p = re.compile("Literal (.*) at line:")
                result = p.search(operator_str)
                right_value = result.group(1)
            elif right_type == 'Constant':
                p = re.compile("Constant (.*) at line:")
                result = p.search(operator_str)
                right_value = result.group(1)
            else:
                raise "Unsupport Value Type: " + right_type
            if re.match(r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$', left_value) is not None:
                left_relation = left_value.split('.')[0]
                left_column = left_value.split('.')[1]
                if left_relation in alias2table:
                    left_relation = alias2table[left_relation]
                left_value = left_relation + '.' + left_column
            else:
                if relation_name is None:
                    relation = index_name.replace(left_value + '_', '')
                else:
                    relation = relation_name
                left_value = relation + '.' + left_value
            if re.match(r'^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$', right_value) is not None:
                right_relation = right_value.split('.')[0]
                right_column = right_value.split('.')[1]
                if right_relation in alias2table:
                    right_relation = alias2table[right_relation]
                right_value = right_relation + '.' + right_column
            sequence.append(Comparison(operator, left_value, right_value.strip('\'')))
            current_line += 1
    return sequence


def pre2seq(predicates, alias2table, relation_name, index_name):
    pr = remove_invalid_tokens(predicates)
    pr = pr.replace("''", " ")
    p = pypred.Predicate(pr)
    try:
        predicates = predicates2seq(p.description().strip('\n').split('\n'), alias2table, relation_name, index_name)
    except:
        raise
    return predicates
