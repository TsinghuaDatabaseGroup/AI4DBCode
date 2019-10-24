#!/bin/sh

cd ../../

SERVER=$(basename `pwd`)
PID_FILE=${SERVER}.pid
DUMMY_FILE=/tmp/DUMMY_${SERVER}
OWN_DUMMY=0

cleanup() {
    if [ $OWN_DUMMY -eq 1 ]; then
        OWN_DUMMY=0
        rm -rf ${DUMMY_FILE}
    fi
}
check_dummy_owner() {
    lock_errno=$?
    if [ $lock_errno -eq 0 ]; then
        OWN_DUMMY=1
    else
        exit $lock_errno
    fi
}

lock_file=/tmp/check_${SERVER}.lock
(
    flock -xn 200
    if [ -r "$DUMMY_FILE" ]; then
        echo "$0 already running"
        exit 1
    fi

    touch $DUMMY_FILE
    rm -rf $lock_file

) 200>$lock_file
check_dummy_owner
trap "cleanup" SIGHUP SIGTERM EXIT

cd ./bin
if [ -f $PID_FILE ];then
        program_pid=`cat $PID_FILE`
        if [ $? -eq 0 ] && [ -n "$program_pid" ];then
                if  ps u -p "$program_pid" | grep -wc ${SERVER} >/dev/null;then
                        echo "[WARN]program is running, quit now"
                        exit 0
                fi
        fi
fi

cd ../tools/cron && ./check_core.sh
if [ $? -eq 3 ];then
        echo "close coredump!!!"
        ulimit -c 0
        export GOTRACEBACK=all
else
        echo "open coredump!!!"
        ulimit -c unlimited
        export GOTRACEBACK=crash
fi
cd -

function shrink_file()
{
        local file=$1
        if [ ! -f $file ];then
                return 1
        fi
        local file_size=`du -b $file | awk '{print $1}'`
        if [ $? -ne 0 ];then
                return 2
        fi
        max_size=32000000  #32M
        if [ $file_size -lt $max_size ];then
                return 3
        fi
        truncate_row=$(wc -l $file | awk '{print $1}')
        ((truncate_row=truncate_row/2))
        if [ $truncate_row -ge 1 ];then
                sed -i "1,${truncate_row}d" $file
                echo "[WARN]Truncate ${truncate_row} rows..."
                return 0
        fi
        return 4
}

LOGFILE="${SERVER}.runlog"
shrink_file ${LOGFILE}

echo "" >> ${LOGFILE}
echo "==================START AT `date +'%Y-%m-%d %H:%M:%M'`==================" >> ${LOGFILE}

nohup ./${SERVER} ../etc/${SERVER}.conf >> ${LOGFILE} 2>&1 &
if [ $? -ne 0 ];then
        echo "[ERROR]start server ${SERVER} failed!"
        exit 1
fi
echo -n $! > $PID_FILE
echo "start ok!"
