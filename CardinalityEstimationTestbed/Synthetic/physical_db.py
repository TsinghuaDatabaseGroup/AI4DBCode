import pandas as pd
import psycopg2


class DBConnection:
    def __init__(self, db_user="jintao", db_password="jintao", db_host="166.111.121.55", db_port="5432",
                 db="benchmark"):
        # def __init__(self, db_user="jintao", db_password="jintao", db_host="166.111.121.62", db_port="5432", db="imdb"):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db = db

    def vacuum(self):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        old_isolation_level = connection.isolation_level
        connection.set_isolation_level(0)
        query = "VACUUM"
        cursor = connection.cursor()
        cursor.execute(query)
        connection.commit()
        connection.set_isolation_level(old_isolation_level)

    def get_dataframe(self, sql):
        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        return pd.read_sql(sql, connection)

    def submit_query(self, sql):
        """Submits query and ignores result."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()

    def get_result(self, sql):
        """Fetches exactly one row of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()
        cursor.execute('set statement_timeout to 18000')  # 
        cursor.execute(sql)
        record = cursor.fetchone()
        result = record[0]

        if connection:
            cursor.close()
            connection.close()

        return result

    def get_result_set(self, sql, return_columns=False):
        """Fetches all rows of result set."""

        connection = psycopg2.connect(user=self.db_user,
                                      password=self.db_password,
                                      host=self.db_host,
                                      port=self.db_port,
                                      database=self.db)
        cursor = connection.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        if connection:
            cursor.close()
            connection.close()

        if return_columns:
            return rows, columns

        return rows


class TrueCardinalityEstimator:
    """Queries the database to return true cardinalities."""

    def __init__(self, db_connection):
        self.db_connection = db_connection

    def true_cardinality(self, query):
        cardinality = self.db_connection.get_result(query)
        return cardinality
