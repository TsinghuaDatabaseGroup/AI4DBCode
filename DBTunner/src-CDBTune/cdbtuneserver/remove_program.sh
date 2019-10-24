#!/bin/sh

if [ $# -ne 1 ];then
    echo "Usage: `basename $0` <program>"
    exit 1
fi
program=$(echo $1 | tr '[:upper:]' '[:lower:]')

echo "remove $program"

rm -f $program package/$program -R

echo "Done."
