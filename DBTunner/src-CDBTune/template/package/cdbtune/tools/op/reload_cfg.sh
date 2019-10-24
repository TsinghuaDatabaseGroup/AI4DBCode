#!/bin/sh

cd ../../

SERVER=$(basename `pwd`)

if [ $# -ge 1 ];then
	pid=$1
	is_server=`ps -lf -p $pid | grep -c $SERVER`
	if [ $is_server -eq 0 ];then
		echo "pid $pid is NOT $SERVER, cannot reload"
		exit 1
	fi
else
	SERVER=$(basename `pwd`)
	PID_FILE=./bin/${SERVER}.pid

	if [ ! -f $PID_FILE ];then
        	echo "$PID_FILE is not exist, $SERVER is not running"
	        exit 1
	fi

	pid=`cat $PID_FILE`
	if [ -z $pid ] || [ $pid -le 1 ];then
        	echo "pid $pid in $PID_FILE is invalid"
	        rm -f $PID_FILE
        	exit 0
	fi
fi
echo "send usr1 signal for reload to $SERVER: $pid"
kill -USR1 $pid
