from src.plan_encoding.encoding_nodes import *


class TreeNode(object):
    def __init__(self, current_vec, parent, idx, level_id):
        self.item = current_vec
        self.idx = idx
        self.level_id = level_id
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

    def get_idx(self):
        return self.idx

    def __str__(self):
        return 'level_id: ' + self.level_id + '; idx: ' + self.idx


def recover_tree(vecs, parent, start_idx):
    if len(vecs) == 0:
        return vecs, start_idx
    if vecs[0] == None:
        return vecs[1:], start_idx + 1
    node = TreeNode(vecs[0], parent, start_idx, -1)
    while True:
        vecs, start_idx = recover_tree(vecs[1:], node, start_idx + 1)
        parent.add_child(node)
        if len(vecs) == 0:
            return vecs, start_idx
        if vecs[0] == None:
            return vecs[1:], start_idx + 1
        node = TreeNode(vecs[0], parent, start_idx, -1)


def dfs_tree_to_level(root, level_id, nodes_by_level):
    root.level_id = level_id
    if len(nodes_by_level) <= level_id:
        nodes_by_level.append([])
    nodes_by_level[level_id].append(root)
    root.idx = len(nodes_by_level[level_id])
    for c in root.get_children():
        dfs_tree_to_level(c, level_id + 1, nodes_by_level)


def encode_plan_job(plan, parameters):
    operators, extra_infos, condition1s, condition2s, samples, condition_masks = [], [], [], [], [], []
    mapping = []

    nodes_by_level = []
    node = TreeNode(plan[0], None, 0, -1)
    recover_tree(plan[1:], node, 1)
    dfs_tree_to_level(node, 0, nodes_by_level)
    #     print (plan)
    #     debug_nodes_by_level(nodes_by_level)
    for level in nodes_by_level:
        operators.append([])
        extra_infos.append([])
        condition1s.append([])
        condition2s.append([])
        samples.append([])
        condition_masks.append([])
        mapping.append([])
        for node in level:
            operator, extra_info, condition1, condition2, sample, condition_mask = encode_node_job(node.item, parameters)
            operators[-1].append(operator)
            extra_infos[-1].append(extra_info)
            condition1s[-1].append(condition1)
            condition2s[-1].append(condition2)
            samples[-1].append(sample)
            condition_masks[-1].append(condition_mask)
            if len(node.children) == 2:
                mapping[-1].append([n.idx for n in node.children])
            elif len(node.children) == 1:
                mapping[-1].append([node.children[0].idx, 0])
            else:
                mapping[-1].append([0, 0])
    return operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping
