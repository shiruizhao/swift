#!/bin/bash
SWIFT_HOME=/volume1/users/szhao/predoc_research/GS_hardware/sw/swift
BLOG_MODELS="survey cancer alarm insurance water hailfinder hepar2 munin"

#cd $BLOG_HOME
#sbt/sbt stage
#cd $SWIFT_HOME 
#make compile 
set -x
echo 'Experiment Results on Patmos:' >> lw_out/experiments_output.csv
for model in $BLOG_MODELS ; do
  cd $SWIFT_HOME
  echo running Swift on $model
  ./lw.sh $model &> lw_out/$model.output
  SWIFT_TIME=$(grep -oP 'running time: \K[[:digit:]]*\.[[:digit:]]*' $model.output)  
  sed -i "$ a\ ${model} ${BLOG_TIME} ${SWIFT_TIME}"  lw_out/experiments_output.csv
done

#./swifty.sh aircraft-static1
#./swifty.sh birthday
#./swifty.sh birthday0
#./swifty.sh burglary
#./swifty.sh csi
#./swifty.sh healthiness
#./swifty.sh hurricane
#./swifty.sh poisson-ball
#./swifty.sh simple-aircraft
#./swifty.sh sybil-attack-large
#./swifty.sh sybil-attack-original
#./swifty.sh tugwar
