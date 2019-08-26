#!/bin/bash
for l in en de
do 
	for f in data/multi30k/*.$l
	do 
		echo $f
		if [[ "$f" != *"test"* ]]
			then sed -i "$ d" $f 
		fi
	done
done