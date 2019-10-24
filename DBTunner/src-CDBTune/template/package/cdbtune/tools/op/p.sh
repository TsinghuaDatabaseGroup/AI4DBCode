#!/bin/sh

cd ../../

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
        exit 2
fi

echo "$SERVER: $pid"
if ps -lfp $pid; then
        exit 0
fi
echo "server is NOT running"
exit 3
