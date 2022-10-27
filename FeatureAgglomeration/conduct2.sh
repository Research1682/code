#!/bin/sh

for fname in 0; do
        python3 ./main_NASA_jm1.py 1 31 $fname
        python3 ./main_NASA_jm1.py 31 61 $fname
        python3 ./main_NASA_jm1.py 61 101 $fname
        wait
done

wait

