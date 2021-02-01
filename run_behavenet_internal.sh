#!/bin/bash

#### Import functions for workflow management 
#### Get the path to this function: 
##execpath="$0"
##echo execpath
##echo "here is the processdir: "
##echo $processdir
##scriptpath="$(dirname "$execpath")/ncap_utils"
##
##source "$scriptpath/workflow.sh"
#### Import functions for data transfer 
##source "$scriptpath/transfer.sh"
##
#### Set up error logging
##errorlog
##
#### Custom setup for this workflow
##source .dlamirc

## Environment setup
##export PATH="/home/ubuntu/anaconda3/bin:$PATH"
source activate behavenet

## Declare local storage locations:
userhome="/media/peter/2TB/john"
datadir="$userhome/neurocaas_data"

## BehaveNet setup
cd "$userhome/neurocaas"
printf "$datadir\n$datadir\n$datadir\n" | ./setup_behavenet.py

## All JSON files in meta.json go in .behavenet
jsonstore=".behavenet"

## Download meta.json first
##aws s3 cp "s3://$bucketname/$configpath/meta.json" "$userhome"

## Parser will return an array of formatted strings representing key-value pairs 
output=$(python meta_parser.py "$userhome/meta.json") 
if [ $? != 0 ];
then
	echo "Error while parsing meta.json, exiting..."
	exit 1
fi 
FILES=($(echo $output | tr -d '[],'))

for file in "${FILES[@]}" ; do
    KEY=${file%:*};
    VAL=${file#*:};
    echo $KEY" XX "$VAL;
done

## Stereotyped download script for data

## Stereotyped download script for config

## Begin BehaveNet model fitting
cd "$userhome/behavenet"
echo "Starting analysis..."

