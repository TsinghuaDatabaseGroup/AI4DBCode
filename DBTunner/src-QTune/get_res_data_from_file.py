def data_from_file(file_path):
    indexs = []
    datas = []
    delaies = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                a = i.split()
                indexs.append(a[0])
                datas.append(a[1])
                delaies.append(a[2])
            f.close()
    except Exception as error:
        print(str(error))

    print(indexs)
    print(datas)
    print(delaies)

    return indexs, datas, delaies


def knob_data_from_file(file_path):
    indexs = []
    datas = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                a = i.split("\t")
                print(a)
                indexs.append(a[0])
                datas.append(a[1])
            f.close()
    except Exception as error:
        print(str(error))

    print(indexs)
    print(datas)

    return indexs, datas


if __name__ == '__main__':
    data_from_file("/Users/Four/Desktop/qtune_results/res_predict-1625452896")
    print("\n\n")
    data_from_file("/Users/Four/Desktop/qtune_results/res_random-1625452896")
    print("\n\n")
