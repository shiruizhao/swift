#!/bin/bash

./swift -e MHSampler -i example/$1.blog -o src/$1.cpp --model-output out

cd src

g++ -Ofast -std=c++11 $1.cpp random/*.cpp -o $1 -L/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/lib -I/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/include -larmadillo

echo "Running "$1
./$1

cd ..
