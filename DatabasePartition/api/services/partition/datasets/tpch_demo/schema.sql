CREATE TABLE SUPPLIER (
        S_SUPPKEY               SERIAL,
        S_NAME                  CHAR(25),
        S_ADDRESS               VARCHAR(40),
        S_NATIONKEY             INTEGER NOT NULL, -- references N_NATIONKEY
        S_PHONE                 CHAR(15),
        S_ACCTBAL               DECIMAL,
        S_COMMENT               VARCHAR(101)
)DISTRIBUTED RANDOMLY;

CREATE TABLE LINEITEM (
        L_ORDERKEY              INTEGER NOT NULL, -- references O_ORDERKEY
        L_PARTKEY               INTEGER NOT NULL, -- references P_PARTKEY (compound fk to PARTSUPP)
        L_SUPPKEY               INTEGER NOT NULL, -- references S_SUPPKEY (compound fk to PARTSUPP)
        L_LINENUMBER    INTEGER,
        L_QUANTITY              DECIMAL,
        L_EXTENDEDPRICE DECIMAL,
        L_DISCOUNT              DECIMAL,
        L_TAX                   DECIMAL,
        L_RETURNFLAG    CHAR(1),
        L_LINESTATUS    CHAR(1),
        L_SHIPDATE              DATE,
        L_COMMITDATE    DATE,
        L_RECEIPTDATE   DATE,
        L_SHIPINSTRUCT  CHAR(25),
        L_SHIPMODE              CHAR(10),
        L_COMMENT               VARCHAR(44)
)DISTRIBUTED RANDOMLY;

CREATE TABLE ORDERS (
        O_ORDERKEY              SERIAL,
        O_CUSTKEY               INTEGER NOT NULL, -- references C_CUSTKEY
        O_ORDERSTATUS   CHAR(1),
        O_TOTALPRICE    DECIMAL,
        O_ORDERDATE             DATE,
        O_ORDERPRIORITY CHAR(15),
        O_CLERK                 CHAR(15),
        O_SHIPPRIORITY  INTEGER,
        O_COMMENT               VARCHAR(79)
)DISTRIBUTED RANDOMLY;

CREATE TABLE CUSTOMER (
        C_CUSTKEY               SERIAL,
        C_NAME                  VARCHAR(25),
        C_ADDRESS               VARCHAR(40),
        C_NATIONKEY             INTEGER NOT NULL, -- references N_NATIONKEY
        C_PHONE                 CHAR(15),
        C_ACCTBAL               DECIMAL,
        C_MKTSEGMENT    CHAR(10),
        C_COMMENT               VARCHAR(117)
)DISTRIBUTED RANDOMLY;