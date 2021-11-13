def get_workload_from_file(file_path):
    workload = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                if i:
                    workload.append(i)
            f.close()
    except Exception as error:
        print(str(error))
    return workload

