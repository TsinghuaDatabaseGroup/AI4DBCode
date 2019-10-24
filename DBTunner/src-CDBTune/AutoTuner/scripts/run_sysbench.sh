#!/usr/bin/env bash

# script_path="/home/rmw/sysbench-1.0/src/lua/"
script_path="/usr/local/sysbench1.0.14/share/sysbench/"

if [ "${1}" == "ro" ]
then
    run_script=${script_path}"oltp_read_only.lua"
elif [ "${1}" == "wo" ]
then
    run_script=${script_path}"oltp_write_only.lua"
else
    run_script=${script_path}"oltp_read_write.lua"
fi

sysbench ${run_script} \
        --mysql-host=$2 \
	--mysql-port=$3 \
	--mysql-user=$4 \
	--mysql-password=$5 \
	--mysql-db=sbtest \
	--db-driver=mysql \
        --mysql-storage-engine=innodb \
        --range-size=100 \
        --events=0 \
        --rand-type=uniform \
	--tables=$6 \
	--table-size=$7 \
	--report-interval=10 \
	--threads=$8 \
	--time=$9 \
	run >> ${10}
