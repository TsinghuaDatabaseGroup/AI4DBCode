#!/bin/sh
 
DEFAULT_ALARM_ID=8693764
if [ $# -ge 1 ];then
	alarm_id=$1
else
	alarm_id=${DEFAULT_ALARM_ID}
fi
 
cd ../../
SERVER=$(basename `pwd`)

core_pattern="/data/coredump/core_%e_%p"
core_prefix="/data/coredump/core_${SERVER}_"

echo "coredump pattern: ${core_pattern}, prefix: ${core_prefix}"
 
echo "${core_pattern}" > /proc/sys/kernel/core_pattern
 
count=`ls -t ${core_prefix}* 2> /dev/null | wc -l`
echo "core files number:${count}"
if [ $count -eq 0 ]; then
         exit 0
fi
 
cd bin && ./alarm "[${SERVER}_alarm]Found $count ${SERVER} coredump files" ${alarm_id}
 
cd -
del_count=`expr ${count} - 3`
if [ $del_count -gt 0 ]; then
	# delete old core files 
	ls -t ${core_prefix}* 2> /dev/null | tail -n ${del_count} | xargs -L 10 rm -f
	exit 3                                                                                                                                                                           
fi
 
exit 0
