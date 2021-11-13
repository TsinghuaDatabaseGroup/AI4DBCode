table_count = 20
rows_count = 10000


def write_insert_to_file(file_path):
    with open(file_path, 'a+') as f:
        for i in range(0, int(rows_count / table_count)):

            for j in range(1, table_count + 1):
                insert = "insert into sbtest{} (k, c, pad) values ((select floor(2+rand()*100000)), '{}', '94657455071-01886877449-66853068383-97480802197-06448926027');".format(
                    j, i + 1)
                f.write(insert)
                f.write('\n')

        f.close()
        print("写入成功")


def write_delete_to_file(file_path):
    with open(file_path, 'a+') as f:
        for i in range(0, int(rows_count / table_count)):
            for j in range(1, table_count + 1):
                delete = "delete from sbtest{} order by id desc limit 1;".format(j)
                f.write(delete)
                f.write('\n')
        f.close()
        print("写入成功")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    write_insert_to_file("/Users/Four/Desktop/insert.txt")
    write_delete_to_file("/Users/Four/Desktop/delete.txt")
