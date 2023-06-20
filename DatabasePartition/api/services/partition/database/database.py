import psycopg2
import json
import re
import os
import paramiko
import time

print(os.getcwd())


# extract columns from the table schema
## read (tables, columns) from the create table file
def table_col(file_name):
    # path = './data/' + file_name + '/tpch-create.sql'
    regex = re.compile(';\($')

    tbl_name = {}
    tbl = ""

    with open(file_name, 'r') as f:
        for line in f.readlines():
            if "CREATE TABLE" in line:
                tbl = line.split()[2]
                tbl_name[tbl.lower()] = []
            elif line != "\n" and ');' not in line and regex.search(line) == None:
                col = line.split()[0]
                tbl_name[tbl.lower()].append(col.lower())

    return tbl_name


def extract_predicates(sql):
    # Remove comments from the SQL statement
    sql = re.sub(r'--.*?\n', '', sql)

    # Remove newlines and multiple spaces from the SQL statement
    sql = re.sub(r'\s+', ' ', sql)

    # Split the SQL statement into subqueries
    subqueries = re.split(r'(?<=\))\s+(?=[A-Za-z]+\s*\()|(?<!\()\s+UNION\s+(?=[A-Za-z]+\s*\()', sql)

    # Extract the filter and join predicates from each subquery
    filter_predicates = []
    join_predicates = []
    for subquery in subqueries:
        # Find the table names and aliases used in the subquery
        tables = {}
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|GROUP BY|HAVING|ORDER BY|\))', subquery, flags=re.IGNORECASE)
        if from_match:
            from_clause = from_match.group(1)
            table_match = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)(?:\s+AS\s+([A-Za-z_][A-Za-z0-9_]*))?', from_clause,
                                     flags=re.IGNORECASE)
            for table_name, alias in table_match:
                tables[table_name] = alias or table_name

        # Find the filter and join predicates in the subquery
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP BY|HAVING|ORDER BY|\)|UNION|\bEXCEPT\b|\bINTERSECT\b|$)',
                                subquery, flags=re.IGNORECASE)
        join_match = re.findall(r'JOIN\s+(.+?)\s+ON\s+(.+?)(?=\s+WHERE\s|$|JOIN\b)', subquery, flags=re.IGNORECASE)
        if where_match:
            filter_predicates.append(where_match.group(1))
        if join_match:
            for join in join_match:
                joins = []
                if 'AND' in join[1]:
                    joins = join[1].split(' AND ')
                elif 'and' in join:
                    joins = join[1].split(' and ')

                if joins == []:
                    join_predicates.append(join[1])
                else:
                    for j in joins:
                        join_predicates.append(j)

        # Replace any aliases in the predicates with the original column names
        for table_name, alias in tables.items():
            if alias:
                for i, predicate in enumerate(filter_predicates + join_predicates):
                    pattern = fr'\b{alias}\.([A-Za-z_][A-Za-z0-9_]*)\b'
                    replacement = f'{table_name}.\\1'
                    predicate = re.sub(pattern, replacement, predicate)
                    if i < len(filter_predicates):
                        filter_predicates[i] = predicate
                    else:
                        join_predicates[i - len(filter_predicates)] = predicate

    return filter_predicates, join_predicates


# connect to a postgresql database, execute the sql (left_table, right_table, predicate) on the database, fetch and output the result
def join_card_on_sampled_data(predicate, left_table, right_table, alias_left_table, alias_right_table, args,
                              sample_size=1):
    # connect to the database
    # print(server,"tpch10default10",db_user,password,db_port)
    conn = psycopg2.connect(host=args.sample_db_host, database=args.sample_db_name, user=args.sample_db_user,
                            password=args.sample_db_password, port=args.sample_db_port)
    cur = conn.cursor()

    # execute the sql
    if alias_left_table != "":
        sql = 'SELECT COUNT(*) FROM {} AS {} JOIN {} AS {} ON {}'.format(left_table, alias_left_table, right_table,
                                                                         alias_right_table, predicate)
    else:
        sql = 'SELECT COUNT(*) FROM {} JOIN {} ON {}'.format(left_table, right_table, predicate)

    print(sql)
    cur.execute(sql)
    result = cur.fetchall()
    # print(result)

    # close the communication with the PostgreSQL database server
    cur.close()
    conn.close()

    return result[0][0]


def execution_under_selected_keys(args, database, partition_keys):
    with open(args.workload_path, 'r') as f:
        # "SELECT * FROM customer c JOIN orders o ON c.c_custkey = o.o_custkey JOIN lineitem l ON o.o_orderkey = l.l_orderkey AND o.o_orderdate = l.l_shipdate WHERE l.l_quantity > 10"
        file_contents = f.read()
        queries = file_contents.split(';')
        workload = [query.strip() for query in queries if query.strip() != ""]

    partition_keys_json = json.dumps(partition_keys)
    partition_keys_json = partition_keys_json.replace('"', '\\"')

    workload_json = json.dumps(workload)
    workload_json = workload_json.replace('"', '\\"')

    # db_password db_user db_host db_port
    source_db_credentials = {
        "host": "127.0.0.1",
        "port": args.db_port,
        "dbname": database,
        "user": args.db_user,
        "password": args.db_password,
    }

    db_credentials_json = json.dumps(source_db_credentials)
    db_credentials_json = db_credentials_json.replace('"', '\\"')

    # schema (with default keys); load data; run queries
    command = "python3 run.py {} \"{}\" {} \"{}\" \"{}\"".format(args.schema, db_credentials_json, database,
                                                                 partition_keys_json, workload_json)
    # command = "python3 run.py {} \"{}\" {} \"{}\" \"{}\" \"{}\"".format(args.schema, db_credentials_json, database,
    #                                                              partition_keys_json, workload_json, args.workload_concurrency)


    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(args.server, username=args.server_username, password=args.server_password, port=args.server_port, timeout=12000)

    # Add keepalive
    transport = ssh.get_transport()
    transport.set_keepalive(30)  # Send a keepalive packet every 60 seconds

    stdin, stdout, stderr = ssh.exec_command("cd {} ; ".format(args.server_script_path) + command)
    output = stdout.readlines()
    ssh.close()

    cnt = 0
    for line in output:
        if "Total run time" in line:
            latency = float(line.split(":")[1].strip())
            cnt = cnt + 1
        if "Througput" in line:
            throughput = float(line.split(":")[1].strip())
            cnt = cnt + 1
    if cnt != 2:
        raise Exception("Error: cannot get the latency and throughput!\n Please check your db cluster..")

    return latency, throughput


def drop_database(args, db_name):
    conn = psycopg2.connect(host=args.db_host, database="postgres", user=args.db_user, password=args.db_password,
                            port=args.db_port)
    cur = conn.cursor()
    conn.autocommit = True

    sql = 'DROP DATABASE IF EXISTS {}'.format(db_name)
    cur.execute(sql)
    cur.close()
    conn.close()


def clone_sample_data_to_database(args):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
    }

    destination_db_credentials = {
        "host": args.sample_db_host,
        "port": args.sample_db_port,
        "dbname": args.sample_db_name,
        "user": args.sample_db_user,
        "password": args.sample_db_password,
    }

    source_conn = psycopg2.connect(**source_db_credentials)
    destination_conn = psycopg2.connect(**destination_db_credentials)

    source_cursor = source_conn.cursor()
    destination_cursor = destination_conn.cursor()

    # Get the list of tables in the source database
    source_cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = source_cursor.fetchall()
    # pdb.set_trace()

    # Clone 0.001% of tuples for each table
    for table in tables:

        table_name = table[0]
        # if table_name in ['part', 'supplier', 'partsupp', 'lineitem', 'customer', 'orders']:
        #     continue

        # Step 1: Copy the schema of the table
        source_cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{}' ORDER BY ordinal_position".format(
                table_name))

        columns = source_cursor.fetchall()

        if not columns:
            print("No columns found for table {}. Skipping...".format(table_name))
            continue

        columns_str = ""
        for i, col in enumerate(columns):
            columns[i] = list(columns[i])
            if "character" in columns[i][1] and "varying" not in columns[i][1]:
                columns[i][1] = "character varying"
            columns_str = columns_str + "{} {}".format(columns[i][0], columns[i][1])
            if i != len(columns) - 1:
                columns_str = columns_str + ', '
                # columns_str = ', '.join(["{} {}".format(col[0], col[1]) for col in columns])

        destination_cursor.execute("CREATE TABLE IF NOT EXISTS {} ({})".format(table_name, columns_str))
        destination_conn.commit()

        # Step 2: Clone 0.01% of tuples
        source_cursor.execute("SELECT * FROM {} WHERE random() < {}".format(table_name, args.sample_ratio))
        rows = source_cursor.fetchall()

        if rows:
            values_placeholder = ', '.join(['%s'] * len(columns))
            insert_query = "INSERT INTO {} VALUES ({})".format(table_name, values_placeholder)
            destination_cursor.executemany(insert_query, rows)
        # pdb.set_trace()

        destination_conn.commit()

    # Commit changes and close connections
    destination_conn.commit()
    source_cursor.close()
    destination_cursor.close()
    source_conn.close()
    destination_conn.close()

    print("finish: load sample data into database!")


def extract_tables(args):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
    }
    conn = psycopg2.connect(**source_db_credentials)
    cursor = conn.cursor()
    # Execute the SQL query to fetch all table names in the sample_data database
    cursor.execute("""
        SELECT table_name, pg_size_pretty(pg_total_relation_size('"' || table_name || '"')) AS table_size
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    table_data = cursor.fetchall()
    cursor.close()
    conn.close()

    # the new database may not have the up-to-date statistics
    time.sleep(2)

    conn2 = psycopg2.connect(**source_db_credentials)
    cursor2 = conn2.cursor()
    # Execute the SQL query to fetch the tuple length for each table
    cursor2.execute("""
        SELECT tablename, avg_width
        FROM pg_stats
        WHERE schemaname = 'public'
    """)

    # Fetch all the tuple lengths
    tuple_data = cursor2.fetchall()

    # Close the cursor and the connection
    cursor2.close()
    conn2.close()

    # Store the table sizes in a dictionary
    table_sizes = {}
    for table_name, table_size in table_data:
        table_sizes[table_name] = table_size

    # Store the tuple lengths in a dictionary
    tuple_lengths = {}
    for table_name, tuple_length in tuple_data:
        if table_name not in tuple_lengths:
            tuple_lengths[table_name] = tuple_length
        else:
            if tuple_length > tuple_lengths[table_name]:
                tuple_lengths[table_name] = tuple_length  # take the max tuple length

    table_info = {}
    table_id = 1
    for table_name in table_sizes.keys():
        if table_name in tuple_lengths:
            table_info[table_name] = {
                'table_id': table_id,
                'table_size': table_sizes[table_name],
                'tuple_length': tuple_lengths[table_name]
            }
        else:
            table_info[table_name] = {
                'table_id': table_id,
                'table_size': table_sizes[table_name],
                'tuple_length': 4
            }
            table_id += 1

    return table_info


def table_statistics(args):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
        "connect_timeout": 1200
    }
    conn = psycopg2.connect(**source_db_credentials)
    cursor = conn.cursor()
    # Execute the SQL query to fetch all table names in the sample_data database
    cursor.execute("""
        SELECT table_name, pg_size_pretty(pg_total_relation_size('"' || table_name || '"')) AS table_size
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    table_data = cursor.fetchall()

    # Store the table sizes in a dictionary
    table_sizes = {}
    column_counts = {}
    for table_name, table_size in table_data:
        table_sizes[table_name] = table_size

        # Execute a SQL query to count the number of columns in the table
        cursor.execute(f"""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name   = '{table_name}'
        """)
        # Fetch the column count
        column_count = cursor.fetchone()[0]
        # Append the table name, table size, and column count to the list
        column_counts[table_name] = column_count

    cursor.close()
    conn.close()

    table_info = []
    table_id = 1
    for table_name in table_sizes.keys():
        table_info.append({
            'table_name': table_name,
            'columns': column_counts[table_name],
            'size': table_sizes[table_name]
        })
        table_id += 1

    return table_info


def distinct_value_ratio(col_name, tbl_name, args):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
    }
    conn = psycopg2.connect(**source_db_credentials)

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # Execute the SQL query to fetch all table names in the sample_data database
    cursor.execute("""
        SELECT count(distinct {}), count(*)
        FROM {}
    """.format(col_name, tbl_name))

    # Fetch all the table names
    table_data = cursor.fetchall()

    # Close the cursor and the connection
    cursor.close()
    conn.close()

    if float(table_data[0][1]) == 0:
        return 0

    return float(table_data[0][0]) / float(table_data[0][1])


def obtain_default_partition_keys(args):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
    }

    source_conn = psycopg2.connect(**source_db_credentials)
    source_cursor = source_conn.cursor()

    # Get the list of tables in the source database
    source_cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    tables = source_cursor.fetchall()

    partition_keys = {}
    for table in tables:
        table_name = table[0]
        partition_keys[table_name] = ""

    return partition_keys


def alter_partition_keys(args, partition_keys):
    source_db_credentials = {
        "host": args.db_host,
        "port": args.db_port,
        "dbname": args.database,
        "user": args.db_user,
        "password": args.db_password,
    }
    conn = psycopg2.connect(**source_db_credentials)

    # Create a cursor object to interact with the database
    cursor = conn.cursor()

    # alter the distribution keys
    for table_name in partition_keys:
        if partition_keys[table_name] == "":
            continue
        cursor = conn.cursor()
        query = "ALTER TABLE {} SET DISTRIBUTED BY ({})".format(table_name, args.partition_keys[table_name])
        cursor.execute(query)
        conn.commit()

    cursor.close()
    conn.close()
