# node type:
import copy
class Node(object):
    def __init__(self, value:int,  l, r, cost, others, oid, sorted_id = 0):
        self.value = value
        self.l = l
        self.r = r
        self.cost = cost
        self.oid = oid
        self.others = others.copy()
        self.sorted_id = sorted_id

def SortL(node:Node):
    return node.l

def merge_op(a:int, b:int):
    c = 0
    if a <= 7:
        c = c | (1 << a)
    else:
        c = a
    if b <= 7:
        c = c | (1 << b)
    else:
        c |= b
    return c

def merge_cost(node_a, node_b):
    node_c = Node(value=-1, oid=node_a.oid, l=-1, r=-1,cost= node_a.cost+node_b.cost,others=[])
    node_c.others.append(merge_op(node_a.others[0],node_b.others[0]))
    node_c.others.append( (node_a.others[1]+node_b.others[1]) )
    node_c.others.append( (node_a.others[2]+node_b.others[2]) )
    node_c.others.append( (node_a.others[3]+node_b.others[3]) )
    return node_c


def Graph_merge(n:int, hash_map:dict, node_list:list):
    # input: n: number of vertices; hash_map: key : (i, j) for all edges; node_list: for all nodes.
    # output: new dict, for the rebuild graph.

    # Step 1 start.
    node_sorted = node_list.copy()
    node_sorted.sort(key = SortL)
    Id2Sorted = {}
    for i in range(n):
        Id2Sorted[node_sorted[i].value] = i
        node_sorted[i].sorted_id = i
    now_set, belong_set = 0, [[node_sorted[0],],]
    now_set_r = node_sorted[0].r
    for i in range(n):
        if i == 0:
            continue
        if node_sorted[i].l <= now_set_r:
            belong_set[now_set].append(node_sorted[i])
        else:
            now_set = now_set + 1
            now_set_r = 0
            belong_set.append([node_sorted[i],])
        now_set_r = max(now_set_r, node_sorted[i].r)
    # Step 1 end.

    # Step 2 begin.
    global_bel_step2,bel_step2, cost = {},{}, {}
    set_step2 = []
    global_set_count = 0
    for setid in range(len(belong_set)):
        cluster = belong_set[setid]
        set_count = 0
        set_step2.append([])
        for _, vi in enumerate(cluster):
            #bitset = []
            #for __ in range(_):
            #    bitset.append(True)
            unvalidset = {}
            merge_front = False
            for __ in range(_):
                vj = cluster[__]
                # print(vi, vj)
                if vj.sorted_id in unvalidset:
                    continue
                if merge_front == True:
                    break
                if (vi.value, vj.value) not in hash_map: # try to merge i into j.
                    merge_flag = True
                    # print("?",bel_step2[vj.sorted_id],"REVEDGE:",vi.value,vj.value)
                    for vk in set_step2[setid][bel_step2[vj.sorted_id]]:
                        if (vk.value, vi.value) in hash_map:
                            merge_flag = False # if merge i into j, will cause self-loop
                    if merge_flag == True:
                        bel_step2[vi.sorted_id] = bel_step2[vj.sorted_id]
                        global_bel_step2[vi.sorted_id] = global_bel_step2[vj.sorted_id]
                        set_step2[setid][bel_step2[vi.sorted_id]].append(vi)
                        cost[global_bel_step2[vi.sorted_id]] = merge_cost(vi, cost[global_bel_step2[vi.sorted_id]])
                        merge_front = True
                    else:
                        for vk in set_step2[setid][bel_step2[vj.sorted_id]]: # Merge i into j is impossible, then merge i into every element in set[j] is impossible.
                                unvalidset[vk.sorted_id] = True

            if merge_front == False: # all nodes can't fulfill merge condition, then add a new set.
                set_count = set_count + 1
                global_set_count += 1
                bel_step2[vi.sorted_id] = set_count - 1
                global_bel_step2[vi.sorted_id] = global_set_count - 1
                set_step2[setid].append([vi, ])
                cost[global_bel_step2[vi.sorted_id]] = copy.deepcopy(vi)
    # Step 2 end.
    # Build new graph.
    new_dict = {}
    max_len = 0
    for key in hash_map:
        a, b = (global_bel_step2[Id2Sorted[key[0]]], global_bel_step2[Id2Sorted[key[1]]])
        a_b_value = hash_map[key]
        # print("edge_new:",a,b,"edge_org",key[0],key[1])
        if (a, b) not in new_dict:  # add edge weight.
            new_dict[(a, b)] = [a_b_value,]
            new_dict[(b, a)] = [a_b_value,]
            max_len = max(max_len, 1)
        else:
            new_dict[(a, b)].append(a_b_value)
            new_dict[(b, a)].append(a_b_value)
            max_len = max(len(new_dict[(a, b)]),max_len)

    ## Divide Ematrix into arrays.
    # Strategy 1. With the order of appending.
    '''
    ematrix_arrays = []
    for i in range(max_len):
        ematrix_temp = []
        for keys in new_dict:
            if len(new_dict[keys]) <= i:
                continue
            element = new_dict[keys][i]
            ematrix_temp.append([keys[0],keys[1],element])
        ematrix_arrays.append(ematrix_temp)
    '''
    # Strategy 2. With Sorted Only.
    '''
    ematrix_arrays = []
    for keys in new_dict:
        new_dict[keys][:] = sorted(new_dict[keys],reverse=True)
    for i in range(max_len):
        ematrix_temp = []
        for keys in new_dict:
            if len(new_dict[keys]) <= i:
                continue
            element = new_dict[keys][i]
            ematrix_temp.append([keys[0], keys[1], element])
        ematrix_arrays.append(ematrix_temp)
    '''
    # Strategy 3. With Sorted & Ones divided(unfinished).
    # '''
    ematrix_arrays = []
    left_element_counts = {}
    for keys in new_dict:
        new_dict[keys][:] = sorted(new_dict[keys],reverse=True)
        left_element_counts[keys] = len(new_dict[keys]) 
    elements_in_ematrix = True
    while elements_in_ematrix:
        ematrix_temp = []
        max_val = -1
        elements_in_ematrix = False
        for keys in new_dict:
            if left_element_counts[keys] == 0:
                continue
            element = new_dict[keys][len(new_dict[keys]) - left_element_counts[keys]]
            max_val = max(max_val, element)
            elements_in_ematrix = True
        if max_val > 0:
            for keys in new_dict:
                if left_element_counts[keys] == 0:
                    continue
                element = new_dict[keys][len(new_dict[keys]) - left_element_counts[keys]]
                if element <= 0:
                    continue
                else:
                    ematrix_temp.append([keys[0], keys[1], element])
                    left_element_counts[keys] -= 1
        else:
            for keys in new_dict:
                if left_element_counts[keys] == 0:
                    continue
                element = new_dict[keys][len(new_dict[keys]) - left_element_counts[keys]]
                ematrix_temp.append([keys[0], keys[1], element])
                left_element_counts[keys] -= 1
        if elements_in_ematrix == True:
            ematrix_arrays.append(ematrix_temp)
    # '''
    # ematrix_arrays format:
    # [ [[a,b,c],[a,b,c],...,[a,b,c]],[[a,b,c],[a,b,c],...], ...]
    return copy.deepcopy(ematrix_arrays), copy.deepcopy(cost)

def mergegraph_main(mergematrix, ematrix, vmatrix):
    n = len(mergematrix)
    m = len(ematrix)
    hash_map = {}
    node_list = []
    for i in range(n):
        oid, run_cost, l, r = mergematrix[i][0],mergematrix[i][4],mergematrix[i][1],mergematrix[i][2]
        others = mergematrix[i][3:] # op. run_cost, float(parent["Actual Startup Time"]), run_time
        # node_feature = [oid, op, run_cost, float(parent["Actual Startup Time"]), run_time]
        value = i
#        print("node:",oid)
#        l, r, cost = map(int, f.readline().split())
        node_list.append(Node(value, l, r, run_cost,others, oid))
    for i in range(m):
        x, y = ematrix[i][0], ematrix[i][1]
        hash_map[(x, y)] = ematrix[i][2]
        hash_map[(y, x)] = ematrix[i][2]
        # print(x, y)

    ematrix_arrays, new_node_all = Graph_merge(n, hash_map, node_list)
    # ematrix = []
    # vmatrix = []
    #for keys in new_graph.keys():
        # print("new_graph:", keys[0], keys[1], new_graph[(keys[0],keys[1])])
    #    ematrix.append([keys[0], keys[1], new_graph[(keys[0],keys[1])]])
#    for node in node_list:
#        print("!", node.cost)

    return new_node_all, ematrix_arrays



if __name__ == '__main__':
    with open("n=500.txt", "r") as f:
        n, m = map(int, f.readline().split())
        hash_map = {}
        node_list = []
        for i in range(n):
            value = i
            l, r, cost = map(int, f.readline().split())
            node_list.append(Node(value,l,r,cost))
        for i in range(m):
            x, y = map(int, f.readline().split())
            hash_map[(x,y)] = True
            hash_map[(y,x)] = True
            # print(x, y)
        new_graph = Graph_merge(n,hash_map,node_list)
        for keys in new_graph.keys():
            print("new_graph:",keys[0], keys[1])
        for node in node_list:
            print("!",node.cost)
