#!/bin/sh

# Usage
# ./reprod_eval.sh ./path/to/tar_models

ROOT_MODEL="$1"

if [ -z "$ROOT_MODEL" ]; then
  echo "Root for model not provided"
  exit -1
fi

if [ ! -d "$ROOT_MODEL" ]; then
  echo "Provided path ${ROOT_MODEL} should be a directory"
  exit -1
fi

echo "Looking through files with '.tar' extensions in your directory."

find $ROOT_MODEL -type f -name "*.tar" | while IFS= read -r model_path 
do
  echo "Found checkpoint $model_path"
  ./experiment.sh jkk
done

