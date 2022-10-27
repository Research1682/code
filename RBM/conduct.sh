#!/bin/sh

bash create_para_dir.sh

python3 ./RBM_PROMISE.py 1 100 &
python3 ./RBM_NASA_PARA.py 1 100 &
python3 ./RBM_AEEEM.py 1 100 &

wait

