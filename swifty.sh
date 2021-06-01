#!/bin/bash

fname=example/$1.blog
fname=${fname##*/}
fname=${fname%.*}

if [[ ! -f example/$1.blog ]] ; then
    echo 'File' example/$1.blog ' is not there, aborting.'
    exit
fi

./swift -i example/$1.blog -o src/$fname.cpp --model-output data

cd src

g++ -Ofast -std=c++11 $fname.cpp random/*.cpp -o $fname -L/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/lib -I/users/micas/szhao/no_backup/software/anaconda3/envs/gem5/include -larmadillo

mv $fname.cpp ../lw_out/$fname.cpp
mv $fname ../lw_out/$fname

echo "Running "$fname
cd ..

./lw_out/$fname
