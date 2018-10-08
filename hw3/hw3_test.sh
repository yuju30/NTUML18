if [ "$3" = "public" ]
then 
	python3 HW3_test.py $1 model_public.hdf5 $2
elif [ "$3" = "private" ]
then 
	python3 HW3_test.py $1 model_private.hdf5 $2
fi

