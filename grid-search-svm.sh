#!/bin/bash

prefix=${1:data/es_train}
nproc=${2:-5}

for min_df in 5 2 1;do 
    for ch in 0 1 2 3 4 5 6 7 8 9;do 
        for w in 0 1 2 3 4;do 
            if [[ $w -eq 0 && $ch -eq 0 ]]; then
                continue
            fi
            for C in 1.0 0.9 1.2 0.5 2.0 0.1 5.0;do
                ./k-fold.py -j $nproc -r $C -f $min_df -C $ch -W $w -i $data
            done
            # Removing the cached data here should not hurt
            rm -f .cache/*
        done
    done
done
