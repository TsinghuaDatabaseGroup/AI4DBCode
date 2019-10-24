#!/bin/sh

cd ../../

SERVER=$(basename `pwd`)
PID_FILE=./bin/${SERVER}.pid
GRACEFUL_CNT=6

if [ ! -f $PID_FILE ];then
        echo "$PID_FILE is not exist, $SERVER is not running"
        exit 0
fi
pid=`cat $PID_FILE`
if [ -z $pid ] || [ $pid -le 1 ];then
        echo "pid $pid in $PID_FILE is invalid"
        rm $PID_FILE
        exit 0
fi

echo "Stopping $SERVER, pid=$pid "

cnt=0
while ps -p $pid >/dev/null 2>&1;
do
        kill $pid 2>/dev/null
        sleep 1
        ((cnt++))
        if [ $cnt -ge $GRACEFUL_CNT ];then
                echo "[WARN]kill $pid gracefully timeout, kill force!"
                kill -9 $pid
                break
        fi
done

rm -f $PID_FILE
echo "stop ok!"
