#!/bin/bash

## Import functions for workflow management 
## Get the path to this function: 
execpath="$0"
echo execpath
echo "here is the processdir: "
echo $processdir
scriptpath="$(dirname "$execpath")/ncap_utils"

source "$scriptpath/workflow.sh"
## Import functions for data transfer 
source "$scriptpath/transfer.sh"

## Set up error logging
errorlog

## Custom setup for this workflow
source .dlamirc

## Environment setup
export PATH="/home/ubuntu/anaconda3/bin:$PATH"
source activate behavenet

## Declare local storage locations:
userhome="/media/peter/2TB/john"
datastore="$userhome/neurocaas_data"
outstore="$userhome/neurocaas_output"

## BehaveNet setup
cd "$userhome/neurocaas"
printf "$datastore\n$outstore\n$outstore\n" | ./setup_behavenet.py

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
    
    file="${file%\'}"
    file="${file#\'}"
    FILETYPE="${file%:*}"
    FILENAME="${file#*:}"

    eval "$FILETYPE=$FILENAME"

    if [[ "$FILETYPE" = "data" ]]
    then
	    ## Stereotyped download script for data
	    aws s3 cp "s3://$bucketname/$inputpath/${file#*:}" "$userhome/$datastore"
	    echo "downloading data $FILENAME"
    else
	    ## Stereotyped download script for config
	    aws s3 cp "s3://$bucketname/$inputpath/${file#*:}" "$userhome/$jsonstore"
	    echo "downloading jsons to .behavenet $FILENAME"
    fi
done

## Begin BehaveNet model fitting
echo "Starting analysis..."
python params_parser.py "$userhome/$jsonstore/$params" "$datastore/$data" "$userhome/$jsonstore/directories.json"
cd "$userhome/behavenet"
RUNCOMMAND="python behavenet/fitting/ae_grid_search.py" 
RUNFLAGS="--data_config $userhome/$jsonstore/$params --model_config $userhome/$jsonstore/$model --training_config $userhome/$jsonstore/$training --compute_config $userhome/$jsonstore/$compute"

eval "$RUNCOMMAND $RUNFLAGS"

echo "Done, uploading results now"

cd "$userhome/$outstore"
aws s3 sync ./ "s3://$bucketname/$groupdir/$processdir"
cd "$userhome"


