# TabularNet

## Structure
├── README.md
│
├── environment.yml # conda environment of TabularNet
│
├── data 
│
├── src
│   ├── encoder # implementation of each part of TabularNet
│   │   ├── bigru.py 
│   │   ├── gine.py   
│   │   ├── rowcolPooling.py
│   │   └── transformer.py
│   │
│   ├── model 
│   │   ├── BaseModel.py  
│   │   └── tabularNet.py 
│   │
│   ├── table2matrix # Processing JSON to handlable data
│   │   ├── cellrole.py
│   │   └── utils.py
│   │
│   └── utils
│       ├── dataset.py # custom dataset and dataloader
│ 
└── train.py

## Training 

python train.py --root_dir ./data/

## Dataset
Dataset is stored at "./data/".

Table is formulated as dict, and their keys and values are listed as follows:
1. "table_feat": [N, M, F], F-dim feature of all the cells
2. "table_label": [N, M], labels of all the cells
3. "table_index": [N * M], index of cells when traverse table by rows. If a subset of cells is merged, then all index for cells in this set is set to be the index of their leader cell.
4. "G_edge": nx.Graph(), The corresponding WordNet tree based graph of each table.

Dataset can be generated from original files, following:
1. Get BERT feature through bert-as-service：
   1. pip install bert-serving-server # server
   2. pip install bert-serving-client # client
   3. download pretrained weights: wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
   4. unzip and then start service：bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4 
   5. Ref.: https://github.com/hanxiao/bert-as-service
2. cd src/table2matrix
3. python cellrole.py # Plz. notice modify the load and save path in this script.



