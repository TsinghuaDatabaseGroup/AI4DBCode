#!/usr/bin/env bash
# run_job.sh  selectedList.txt  queries_dir   output

avg_lat=0
avg_tps=0
count=0

printf "query\tlat(ms)\n" > $7

tmp_fifofile="/tmp/$$.fifo"
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile

thread_num=$8

rm "/tmp/avg_lat_pipef.txt"
rm "/tmp/avg_tps_pipef.txt"
rm "/tmp/count_pipef.txt"

for ((i=0;i<${thread_num};i++));do
    echo
done >&6

for file in $6/*;
do
    read -u6
    {
        tmp=$(mysql -h$1 -p$2 -u$3 -p$4 $5 < $file | tail -n 1 )
        query=`echo $tmp | awk '{print $1}'`
        lat=`echo $tmp | awk '{print $2}'`
        mysql -h$1 -p$2 -u$3 -p$4 $5 -e"\q"
        tps=$(echo "scale=4; 60000 / $lat" | bc)

        echo "scale=4; $lat / 1000" | bc >> "/tmp/avg_lat_pipef.txt"
        echo "scale=4; $tps" | bc >> "/tmp/avg_tps_pipef.txt"
        echo "1" | bc >> "/tmp/count_pipef.txt"
        printf "$query\t$lat\n" >> $7
        echo >&6
    } &
done
wait
exec 6>&-

avg_lat=$(echo $(echo -n `cat /tmp/avg_lat_pipef.txt | awk '{print $1}'`| tr ' ' '+')|bc)
avg_tps=$(echo $(echo -n `cat /tmp/avg_tps_pipef.txt | awk '{print $1}'`| tr ' ' '+')|bc)
count=$(echo $(echo -n `cat /tmp/count_pipef.txt | awk '{print $1}'`| tr ' ' '+')|bc)

rm "/tmp/avg_lat_pipef.txt"
rm "/tmp/avg_tps_pipef.txt"
rm "/tmp/count_pipef.txt"

avg_lat=`echo "scale=4; $avg_lat/$count" | bc`
avg_tps=`echo "scale=4; $avg_tps/$count" | bc`

printf "\navg_tps(txn/min): \t%5.4f\navg_lat(ms): \t%5.4f\n" $avg_tps $avg_lat >> $7
