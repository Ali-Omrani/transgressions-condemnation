#!/bin/bash
for i in {0..19}
do
   #echo "user_ids_5mil_7days_part_$i.csv"
   Rscript ./HPC_follower_crawl.R ../../../data/user_ids/user_ids_5mil_7days_part_$i.csv > ../outputs/out$i.txt &
	echo "alaki"
done


#for file in ../../data/user_ids/*
#do
#	Rscript HPC_follower_crawl.R ../../data/user_ids/user_ids_5mil_7days_part_$i.csv > out$i.txt &

#done

