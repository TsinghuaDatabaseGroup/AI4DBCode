#!/bin/sh
#version: 1

function del_program_from_crontab()
{
    program=$1

    program_service=`crontab -l|grep "#for $program"`
    if [ "${program_service}" == "" ]; then #already deleted
        return
    fi

    crontab_file=/tmp/unistall_${program}_crontab.txt
    crontab -l > $crontab_file

    sed -i "/#for ${program}/{N;d}" $crontab_file

    crontab $crontab_file
    rm -f $crontab_file
}

function check_process()
{
    for ((i=0;i<10;i++))
    do
        sleep 1
        if ! ./p.sh >/dev/null;then
		echo "process is stopped..."
		return 0
	fi
    done

    return 1 
}


BASE_DIR=`cd ../..; pwd`
PROGRAM_NAME=`basename $BASE_DIR`

echo "del crontab..."
del_program_from_crontab "$PROGRAM_NAME"

echo "try to stop ${PROGRAM_NAME}"
sleep 1

i=0
while true
do
	./stop.sh >/dev/null 2>&1

	check_process
	if [ $? -eq 0 ];then
		exit 0
	fi
	((i++))
	if [ $i -gt 3 ];then
		echo "stop process failed!"
		exit 1
	fi
	echo "process is running,try to stop"
done

