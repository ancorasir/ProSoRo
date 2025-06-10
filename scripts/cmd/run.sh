#!/bin/bash

pasway=`dirname $0`
pushd $pasway
echo $basename
for file in `ls $pasway/*.inp`;
do
abq job=`basename $file` cpus=4 int;
done
echo all finished
read -n 1