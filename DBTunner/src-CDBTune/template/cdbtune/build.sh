#!/bin/sh

PROGRAM=$(basename `pwd`)
DESTBIN=../package/${PROGRAM}/bin/

echo "begin to build for debug..."
go build  -gcflags "-N -l"

if [ $? -eq 0 ];then
	echo "mv $PROGRAM to $DESTBIN"
	mv $PROGRAM $DESTBIN
fi
