from api.services.partition.config import PartitionConfig
from api.services.partition.partition_selection.partition_key_selection import partition_key_selection


if __name__ == "__main__":
    # 生成参数
    args = PartitionConfig()
    # 生成路径
    success, msg = args.generate_paths()
    if not success:
        raise ValueError(msg)
    partition_key_selection(args)