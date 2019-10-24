#!/bin/sh
#version: 1

function add_program_to_crontab()
{
    program_to_add=$1
    cmd_in_crontab=$2

    program_service=`crontab -l|grep "^#for $program_to_add"`
    if [ "${program_service}" != "" ]; then #already add
        return
    fi

    cd /usr/local/
    crontab -l > tmp_crontab.txt

    echo "" >> tmp_crontab.txt
    echo "#for $program_to_add" >> tmp_crontab.txt
    echo "$cmd_in_crontab" >> tmp_crontab.txt

    crontab tmp_crontab.txt
    rm -f tmp_crontab.txt
}

function check_process()
{
    for ((i=0;i<10;i++))
    do
        sleep 1
        if ./p.sh >/dev/null;then
                echo "process is running now"
                return 1
        fi
    done
    return 0
}


echo "starting server..."
./start.sh >/dev/null 2>&1

check_process
if [ $? -eq 0 ];then
    echo "some process launch failed!"
    exit 1;
fi

BASE_DIR=`cd ../..; pwd`
PROGRAM_NAME=`basename $BASE_DIR`

echo "add crontab..."
add_program_to_crontab "$PROGRAM_NAME" "* * * * * cd $BASE_DIR/tools/cron && ./check_all.sh > /dev/null 2>&1"
