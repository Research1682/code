#!/bin/sh

for fname in `seq 0 9`; do
        python3 ./main_PROMISE.py 1 101 $fname &
done
wait

