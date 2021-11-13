def get_workload(file_path):
    workload = []
    try:
        with open(file_path, 'r') as f:
            result = f.read().splitlines()
            for i in result:
                if ("#" not in i) and ("SET timestamp" not in i) and ("COMMIT" not in i) and ("BEGIN" not in i) and i:
                    workload.append(i)
            f.close()
        print(f"共有{len(workload)}条数据")
        with open("/Users/Four/Desktop/workload123.txt", "w+") as sql_file:
            for query in workload:
                if query:
                    sql_file.write(query + "\n")
            sql_file.close()
    except Exception as error:
        print(str(error))


if __name__ == '__main__':

    # get_workload("/Users/Four/Desktop/slow.log")

    for i in range(0, 1):

        for j in range(1, 21):
            insert = "insert into sbtest{} (k, c, pad) values ((select floor(2+rand()*100000)), '{}', '94657455071-01886877449-66853068383-97480802197-06448926027');".format(
                j, i + 1)
            print(insert)
