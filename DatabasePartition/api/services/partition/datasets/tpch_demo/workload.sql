SELECT * FROM customer c 
JOIN orders o ON c.c_custkey = o.o_custkey 
JOIN lineitem l ON o.o_orderkey = l.l_orderkey AND o.o_orderdate = l.l_shipdate 
WHERE l.l_quantity > 10;