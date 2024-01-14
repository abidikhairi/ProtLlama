#!/bin/bash

pip install wandb transformers kaggle

cp ./.secrets/kaggle.json ~/.kaggle/kaggle.json

mkdir data && kaggle datasets download -d abidikhairi/uniport-reviewed
unzip uniport-reviewed.zip
