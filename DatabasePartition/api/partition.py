import logging
from flask import Blueprint, request
from api.utils.code import ResponseCode
from api.utils.response import ResMsg
from api.utils.util import route, get_dataset_names

from api.services.partition.partition_selection.partition_key_selection import partition_key_selection
from api.services.partition.database import *
from api.services.partition.partition_selection.selection_model import *
from api.services.partition.partition_evaluation.evaluation_model import *
from api.services.partition.config import PartitionConfig


bp = Blueprint("partition", __name__, url_prefix='/partition')

logger = logging.getLogger(__name__)


@route(bp, '/dataset', methods=["POST"])
def dataset():
    """
    return existing datasets in the program (./api/services/partition/datasets/tpch_demo)
    :return:
    """
    res = ResMsg()

    # find existing datasets
    datasets = get_dataset_names()
    res.update(data=datasets)
    return res.data


@route(bp, '/distribution', methods=["POST"])
def distribution():
    """
    return the data and workload statistics of the dataset
    :return:
    """
    res = ResMsg()
    obj = request.get_json(force=True)
    dataset_name = obj.get("dataset")
    # 未获取到参数或参数不存在
    if not obj or not dataset_name:
        res.update(code=ResponseCode.InvalidParameter)
        return res.data

    # 生成参数
    args = PartitionConfig()
    args.database = dataset_name
    # 生成路径
    success, msg = args.generate_paths()
    # 生成路径失败
    if not success:
        res.update(code=ResponseCode.InvalidParameter, msg=msg)
        return res.data


    # obtain the table info
    tbls = table_statistics(args)
    # {'lineitem': ['l_quantity', 'l_shipdate'], 'orders': ['o_orderkey', 'o_custkey'], 'customer': ['c_custkey']}

    print("tbls: ", tbls)

    # obtain the column info (in column graph)
    graph = Column2Graph(args)

    # todo：graph.vertex_matrix graph.edge_matrix 取值，需要转变成 json 识别的格式
    # json format

    # 数据集获取
    data = {
        "distributions": tbls,
        "column2Graph": [graph.vertex_json, graph.edge_json]
    }
    res.update(data=data)
    return res.data


@route(bp, '/recommend', methods=["POST"])
def recommend():
    """
    return recommended partition keys
    :return:
    """

    res = ResMsg()
    obj = request.get_json(force=True)
    dataset_name = obj.get("dataset")
    # 未获取到参数或参数不存在
    if not obj or not dataset_name:
        res.update(code=ResponseCode.InvalidParameter)
        return res.data
    # 生成参数
    args = PartitionConfig()
    args.database = dataset_name
    # 生成路径
    success, msg = args.generate_paths()
    # 生成路径失败
    if not success:
        res.update(code=ResponseCode.InvalidParameter, msg=msg)
        return res.data

    partition_keys = partition_key_selection(args)

    data = []
    for table, columns in partition_keys.items():
        data.append({"table": table, "column": ','.join(columns), "partition_sql": f"ALTER TABLE {table} set distributed by({','.join(columns)});"})

    # 数据集获取
    res.update(data=data)
    return res.data


@route(bp, '/preview', methods=["POST"])
def preview():
    """
    执行preview，并返回执行结果
    :return:
    """

    # "partition_keys": {
    #     "lineitem": "l_orderkey, l_quantity",
    #     "orders": "o_orderdate"
    # }

    res = ResMsg()
    obj = request.get_json(force=True)
    partition_keys = obj.get("partition_keys")
    # 未获取到参数或参数不存在
    if not obj or not partition_keys or partition_keys == {}:
        res.update(code=ResponseCode.InvalidParameter)
        return res.data

    # 生成参数
    args = PartitionConfig()
    # 生成路径
    success, msg = args.generate_paths()
    # 生成路径失败
    if not success:
        res.update(code=ResponseCode.InvalidParameter, msg=msg)
        return res.data

    args.partition_keys = partition_keys

    partition_eval = partition_evaluation_model(args)

    # estimated latency and data distribution
    origin_partition_keys = database.obtain_default_partition_keys(args)
    database.alter_partition_keys(args, partition_keys)
    origin_partitioned_sample_graph = SampleGraph(args, partitioning_keys=origin_partition_keys)
    partitioned_sample_graph = SampleGraph(args, partitioning_keys=partition_keys)

    if args.use_estimated_results:
        origin_latency = partition_eval.estimate_latency(partition_eval.embedding(origin_partitioned_sample_graph))
        latency = partition_eval.estimate_latency(partition_eval.embedding(partitioned_sample_graph))
    else:
        origin_latency, real_origin_throughput = database.execution_under_selected_keys(args, args.database+"_tmp", origin_partition_keys)
        latency, real_throughput = database.execution_under_selected_keys(args, args.database+"_tmp2", partition_keys)
        database.drop_database(args, args.database+"_tmp")
        database.drop_database(args, args.database+"_tmp2")

    data = {
        "performance": {
            "before": origin_latency.item(),
            "after": latency.item()
        },
        "distribution": {
            "before": origin_partitioned_sample_graph.vertex_json,
            "after": partitioned_sample_graph.vertex_json
        }
    }
    res.update(data=data)
    return res.data