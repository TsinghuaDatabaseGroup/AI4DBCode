cd ../../

SERVER=$(basename `pwd`)
PID_FILE=./bin/${SERVER}.pid

if [ -f $PID_FILE ];then
        pid=`cat $PID_FILE`
        if [ -n $pid ] && ps -p $pid >/dev/null;then
                echo "${SERVER} pid $pid is still exist..."
                exit 0
        fi
        echo "pid $pid has gone, restart now"
        rm -f $PID_FILE
else
        echo "$PID_FILE is not exist, $SERVER is not running"
fi

cd ./tools/op && ./start.sh
if [ $? -ne 0 ];then
        echo "[ERROR]restart failed!"
        exit 1
fi
