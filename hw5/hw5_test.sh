#!/bin/sh

if [ "$3" = "public" ]
then 
	python3 HW5_test.py $1 $2 model.hdf5 token
elif [ "$3" = "private" ]
then 
	python3 HW5_test.py $1 $2 model.hdf5 token
fi

