import os
import json
import time

mp_optype = {'Aggregate': 0, 'Nested Loop': 1, 'Index Scan': 2, 'Hash Join': 3, 'Seq Scan': 4, 'Hash': 5, 'Update': 6}


# actual runtime:  actuall executed (training data) / estimated by our model
# operators in the same plan can have data conflicts (parallel)
def compute_cost(node):
    return (float(node["Total Cost"]) - float(node["Startup Cost"])) / 1e6


def compute_time(node):
    # return float(node["Actual Total Time"]) - float(node["Actual Startup Time"])
    return float(node["Actual Total Time"])  # mechanism within pg


def get_used_tables(node):
    tables = []

    stack = [node]
    while stack != []:
        parent = stack.pop(0)

        if "Relation Name" in parent:
            tables.append(parent["Relation Name"])

        if "Plans" in parent:
            for n in parent["Plans"]:
                stack.append(n)

    return tables

def extract_plan(sample, conflict_operators, oid, min_timestamp):
    # from performance_graphembedding_checkpoint import oid
    # from performance_graphembedding_checkpoint import min_timestamp
    if min_timestamp < 0:
        min_timestamp = float(sample["start_time"])
        start_time = 0
    else:
        start_time = float(sample["start_time"]) - min_timestamp
    # function: extract SQL feature
    # return: start_time, node feature, edge feature

    plan = sample["plan"]
    while isinstance(plan, list):
        plan = plan[0]
    # Features: print(plan.keys())
    # start time = plan["start_time"]
    # node feature = [Node Type, Total Cost:: Actual Total Time]
    # node label = [Actual Startup Time, Actual Total Time]

    plan = plan["Plan"]  # root node
    node_matrix = []
    edge_matrix = []
    node_merge_matrix = []

    # add oid for each operator
    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        parent["oid"] = oid
        oid = oid + 1

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)

    stack = [plan]
    while stack != []:
        parent = stack.pop(0)
        run_cost = compute_cost(parent)
        run_time = compute_time(parent)
        # print(parent["Actual Total Time"], parent["Actual Startup Time"], run_time)

        if parent["Node Type"] not in mp_optype:
            mp_optype[parent["Node Type"]] = len(mp_optype)

        tables = get_used_tables(parent)
        # print("[tables]", tables)

        operator_info = [parent["oid"], start_time + parent["Startup Cost"] / 1e6,
                         start_time + parent["Total Cost"] / 1e6]

        for table in tables:
            if table not in conflict_operators:
                conflict_operators[table] = [operator_info]
            else:
                conflict_operators[table].append(operator_info)

        node_feature = [parent["oid"], mp_optype[parent["Node Type"]], run_cost,
                        start_time + float(parent["Actual Startup Time"]), run_time]

        node_matrix = [node_feature] + node_matrix

        node_merge_feature = [parent["oid"], start_time + parent["Startup Cost"] / 1e6,
                              start_time + parent["Total Cost"] / 1e6, mp_optype[parent["Node Type"]], run_cost,
                              start_time + float(parent["Actual Startup Time"]), run_time]
        node_merge_matrix = [node_merge_feature] + node_merge_matrix

        if "Plans" in parent:
            for node in parent["Plans"]:
                stack.append(node)
                edge_matrix = [[node["oid"], parent["oid"], 1]] + edge_matrix

    # node: 18 * featuers
    # edge: 18 * 18

    return start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix, min_timestamp

def overlap(node_i, node_j):
    if (node_j[1] < node_i[2] and node_i[2] < node_j[2]):

        return (node_i[2] - node_j[1]) / (node_j[2] - min(node_i[1], node_j[1]))

    elif (node_i[1] < node_j[2] and node_j[2] < node_i[2]):

        return (node_j[2] - node_i[1]) / (node_i[2] - min(node_i[1], node_j[1]))

    else:
        return 0

def add_across_plan_relations(conflict_operators, knobs, ematrix):
    # TODO better implementation
    data_weight = 0.1
    for knob in knobs:
        data_weight *= knob
    # print(conflict_operators)

    # add relations [rw/ww, rr, config]
    for table in conflict_operators:
        for i in range(len(conflict_operators[table])):
            for j in range(i + 1, len(conflict_operators[table])):

                node_i = conflict_operators[table][i]
                node_j = conflict_operators[table][j]

                time_overlap = overlap(node_i, node_j)
                if time_overlap:
                    ematrix = ematrix + [[node_i[0], node_j[0], -data_weight * time_overlap]]
                    ematrix = ematrix + [[node_j[0], node_i[0], -data_weight * time_overlap]]

                '''
                if overlap(i, j) and ("rw" or "ww"):
                    ematrix = ematrix + [[conflict_operators[table][i], conflict_operators[table][j], data_weight * time_overlap]]
                    ematrix = ematrix + [[conflict_operators[table][j], conflict_operators[table][i], data_weight * time_overlap]]
                '''

    return ematrix


def generate_graph(wid, path):
    # global oid, min_timestamp
    from performance_graphembedding_checkpoint import Database
    # from performance_graphembedding_checkpoint import oid
    # from performance_graphembedding_checkpoint import min_timestamp
    # fuction
    # return
    # todo: timestamp

    vmatrix = []
    ematrix = []
    mergematrix = []
    conflict_operators = {}

    oid = 0
    min_timestamp = -1
    with open( os.path.join(path,"sample-plan-" + str(wid) + ".txt"), "r") as f:
        # vertex: operators
        # edge: child-parent relations
        for sample in f.readlines():
            sample = json.loads(sample)

            # Step 1: read (operators, parent-child edges) in separate plans
            start_time, node_matrix, edge_matrix, conflict_operators, node_merge_matrix, min_timestamp = extract_plan(sample,
                                                                                                       conflict_operators, oid, min_timestamp)

            mergematrix = mergematrix + node_merge_matrix
            vmatrix = vmatrix + node_matrix
            ematrix = ematrix + edge_matrix

        # Step 2: read related knobs
        db = Database("mysql")
        knobs = db.fetch_knob()

        # Step 3: add relations across queries
        ematrix = add_across_plan_relations(conflict_operators, knobs, ematrix)

        # edge: data relations based on (access tables, related knob values)
    return vmatrix, ematrix, mergematrix, oid, min_timestamp
def output_file():
    from performance_graphembedding_checkpoint import data_path
    start_time = time.time()
    num_graphs = 3000
    # notation: oid may be unused.
    for wid in range(num_graphs):
        st = time.time()
        vmatrix, ematrix, mergematrix, oid, min_timestamp = generate_graph(wid, data_path)
        # optional: merge
        # vmatrix, ematrix = merge.mergegraph_main(mergematrix, ematrix, vmatrix)
        print("[graph {}]".format(wid),
              "time:{}; #-vertex:{}, #-edge:{}".format(time.time() - st, len(vmatrix), len(ematrix)))

        with open(os.path.join(data_path, "graph", "sample-plan-" + str(wid) + ".content"), "w") as wf:
            for v in vmatrix:
                wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
        with open(os.path.join(data_path, "graph", "sample-plan-" + str(wid) + ".cites"), "w") as wf:
            for e in ematrix:
                wf.write(str(e[0]) + "\t" + str(e[1]) + "\t" + str(e[2]) + "\n")

    end_time = time.time()
    print("Total Time:{}".format(end_time - start_time))
