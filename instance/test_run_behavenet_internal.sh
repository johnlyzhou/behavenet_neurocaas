#!/bin/bash

## Declare local storage locations:
userhome="/home/ubuntu"
datastore="$userhome/neurocaas_data"
outstore="$userhome/neurocaas_output"

## Move into script directory
cd "$userhome/neurocaas_remote"

## All JSON files in config go in .behavenet
jsonstore="$userhome/.behavenet"

## Download config file first
#aws s3 cp "s3://$bucketname/$configpath" "$userhome"

## Parser will return an array of formatted strings representing key-value pairs
output=$(python ncaasconfig_parse.py "$userhome/config.json")

if [ $? != 0 ];
then
	echo "Error while parsing config, exiting..."
	exit 1
fi

FILES=($(echo $output | tr -d '[],'))

for file in "${FILES[@]}" ; do

    file="${file%\'}"
    file="${file#\'}"
    FILETYPE="${file%:*}"
    FILENAME="${file#*:}"

    eval "$FILETYPE=$FILENAME"

    if [[ "$FILETYPE" = "hdf5" ] || [ "$FILETYPE" = "video" ] || [ "$FILETYPE" = "markers" ]]
    then
	    ## Stereotyped download script for data
	    #aws s3 cp "s3://$bucketname/$(dirname "$inputpath")/${file#*:}" "$datastore"
	    echo "Downloading data $FILENAME"
    else
	    ## Stereotyped download script for jsons
	    #aws s3 cp "s3://$bucketname/$(dirname "$inputpath")/${file#*:}" "$jsonstore"
	    echo "Downloading json $FILENAME"
    fi

done

## Begin BehaveNet analysis
echo "File downloads complete, beginning analysis..."
output=$(python parameter_parse.py "$jsonstore/$params" "$datastore/$data" "$jsonstore/directories.json")

if [ $? != 0 ];
then
	echo "Error while parsing $params, exiting..."
	exit 1
fi

cd "$userhome/behavenet"

echo behavenet/fitting/ae_grid_search.py --data_config "$jsonstore/$params" --model_config "$jsonstore/$model" --training_config "$jsonstore/$training" --compute_config "$jsonstore/$compute"
