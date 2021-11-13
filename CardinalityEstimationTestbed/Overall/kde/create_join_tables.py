from itertools import combinations


def create_joins_script_generate(join_sample_dir, database):
    alias = ['ci', 'mc', 'mi', 'mi_idx', 'mk']
    sqls = []
    for num in range(1, 6):
        for p in combinations(alias, num):
            key = '_'.join(sorted([x for x in p] + ['t']))
            with open('{}/{}.csv'.format(join_sample_dir, key), 'r') as f:
                head = f.readline().strip()
                columns = [x.replace(':', '_') for x in head.split(',')]
            sqls.append('CREATE TABLE IF NOT EXISTS {} ({});'.format(key, ','.join([x + ' FLOAT' for x in columns])))
            sqls.append("\copy {} FROM '{}' CSV HEADER;".format(key, '{}/{}.csv'.format(join_sample_dir, key)))
    with open('script.sql', 'w') as f:
        for sql in sqls:
            f.write(sql)
            f.write('\n')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create Join Samples.')
    parser.add_argument('--join-sample-dir', type=str)
    parser.add_argument('--database', type=str)
    args = parser.parse_args()
    create_joins_script_generate(args.join_sample_dir, args.database)
