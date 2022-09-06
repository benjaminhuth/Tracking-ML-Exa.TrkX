#!/bin/bash

# First copy the files to the corresponding module dirs
cp configs/embedding.yaml LightningModules/Embedding/embedding.yaml
cp configs/filter.yaml LightningModules/Filter/filter.yaml
cp configs/gnn.yaml LightningModules/GNN/gnn.yaml
cp configs/processing.yaml LightningModules/ProcessingODD/processing.yaml

export PYTHONWARNINGS="ignore"

export EXATRKX_DATA=/home/iwsatlas1/bhuth/exatrkx/data_odd/test_data

export PROCESSING_OUTPUT=tmp/processing_output
export EMBEDDING_OUTPUT=tmp/embedding_output
export FILTER_OUTPUT=tmp/filter_output
export GNN_OUTPUT=tmp/gnn_output
export SEGMENTING_OUTPUT=tmp/segmenting_output

export PROJECT_NAME="ODD-1k-no-cut"

traintrack "$@" configs/pipeline.yaml

# Finally remove all yaml files
find ./LightningModules -name "*.yaml" -type f -print0 | xargs -0 /bin/rm -f
