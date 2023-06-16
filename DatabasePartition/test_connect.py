from api.services.partition.config import PartitionConfig


# evaluate the query latency under the selected partitioning keys
from api.services.partition.database import table_statistics

if __name__ == "__main__":
    args = PartitionConfig()
    print("args111: ", args)
    args.database = "tpch_demo"
    # obtain the table info
    tbls = table_statistics(args)

    print(tbls)