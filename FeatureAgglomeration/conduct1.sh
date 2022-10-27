#!/bin/sh

for fname in `seq 0 4`; do
        python3 ./main_AEEEM.py 1 101 $fname &
        wait
done

for fname in `seq 0 9`; do
        python3 ./main_NASA.py 1 101 $fname &
        wait
done

wait

