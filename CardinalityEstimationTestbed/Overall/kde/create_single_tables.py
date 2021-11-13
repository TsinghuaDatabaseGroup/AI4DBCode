def create_single_script_generate(file_path, database, table_name):
    sqls = []
    sqls.append(
        'CREATE TABLE IF NOT EXISTS {} ({});'.format(table_name, ','.join([x + ' FLOAT' for x in ['col0', 'col1']])))
    sqls.append("\copy {} FROM '{}' CSV HEADER;".format(table_name, '{}'.format(file_path)))
    with open('script.sql', 'w') as f:
        for sql in sqls:
            f.write(sql)
            f.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Join Samples.')
    parser.add_argument('--path', type=str)
    parser.add_argument('--database', type=str)
    parser.add_argument('--table-name', type=str)
    args = parser.parse_args()
    create_single_script_generate(args.path, args.database, args.table_name)
