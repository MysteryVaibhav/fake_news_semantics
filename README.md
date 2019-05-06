# Using Semantics to Understand Fake News
Attempt to identify / use semantics in fake news using deep learning techniques.

Make sure the following files are present as per the directory structure before running the code,
```
fake_news_semantics
│   README.md
│   *.py
│   
└───data
    │   balancedtest.csv
    │   fulltrain.csv
    |   test.xsls
```

Dependencies,
```
pytorch 1.0.0
pandas
tqdm
xlrd (pip install xlrd)
bert-pytorch (pip install pytorch-pretrained-bert)
```

To train a LSTM model, run the following command,
```
python main.py --batch_size 1024 --config lstm --encoder 0 --mode 0
```

To train a CNN model, run the following command,
```
python main.py --batch_size 1024 --config cnn --encoder 1 --mode 0
```

To train a BERT model, run the following command,
```
python bert_classifier.py --batch_size 4 --max_epochs 10 --max_seq_length 500 --max_sent_length 70 --mode 0
```

To train a GCN based model, run the following command,
```
python main.py --batch_size 32 --max_epochs 10 --config gcn --max_sent_len 50 --encoder 2 --mode 0
```

To train a GCN based model with attention, run the following command,
```
python main.py --batch_size 32 --max_epochs 10 --config gcn_attn --max_sent_len 50 --encoder 3 --mode 0
```

To train a GATconv based model, run the following command,
```
python main.py --batch_size 32 --max_epochs 10 --config gat --max_sent_len 50 --encoder 4 --mode 0
```

To test the accuracy of the model on the out of domain test set, run the following command,

For the LSTM model,
```
python main.py --batch_size 1024 --encoder 0 --model_file model_lstm.t7 --mode 1
```

For the CNN model,
```
python main.py --batch_size 1024 --encoder 1 --model_file model_cnn.t7 --mode 1
```

For the Bert model,
```
python bert_classifier.py --batch_size 4 --model_file model_bert.t7 --max_seq_length 500 --max_sent_length 70 --mode 1
```

For the GCN model,
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 2 --model_file model_gcn.t7 --mode 1
```

For the GCN model with attention,
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 3 --model_file model_gcn_attn.t7 --mode 1
```

For the GATconv model,
```
python main.py --batch_size 32 --max_sent_len 50 --encoder 4 --model_file model_gat_attn.t7 --mode 1
```

## Baseline Results

### In domain test set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 92.3 | | |
LSTM | 91.4 | | | 
BERT | 88.0 | | | 
LSTM + GCN + Max Pool | 92.8 | | | 
LSTM + GCN + Max Pool + Semantic Adj | 92.7 | | | 
LSTM + GCN + Attn | 93.6 | | |
LSTM + GCN + Attn + Semantic Adj| 93.3 | | |
LSTM + GAT | 93.4 | | |
LSTM + GAT + Semantic Adj | 93.7 | | |
LSTM + GAT + 2 Attn Heads| 93.5 | | |
LSTM + GAT + 2 Attn Heads + Semantic Adj | 92.5 | | |

### Out of domain test set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 67.5 | 67.5 | 67.5 | 67.4
LSTM | 81.4 | 82.2 | 81.4 | 81.3
BERT | 78.1 | 78.1 | 78.1 | 78.0
LSTM + GCN + Max Pool | 85.0 | 85.9 | 85.0 | 85.1
LSTM + GCN + Max Pool + Semantic Adj | 86.4 | 86.4 | 86.3 | 86.4
LSTM + GCN + Self Attn | 86.6 | 87.1 | 86.9 | 86.9
LSTM + GCN + Self Attn + Semantic Adj | 87.8 | 87.8 | 87.8 | 87.8
LSTM + GAT | 86.1 | 86.2 | 86.1 | 86.1 
LSTM + GAT + Semantic Adj | 87.5 | 87.5 | 87.5 | 87.4 
LSTM + GAT + 2 Attn Heads| 88.6 | 89.1 | 88.9 | 88.9
LSTM + GAT + 2 Attn Heads + Semantic Adj | 84.7 | 85.2 | 84.7 | 84.6 
SoTA | - | 90.0 | 84.0 | 87.0

### Results with a dev/test split based on news sources: This might be a more realistic split

### For two classes Satire / Trusted

### In domain dev set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 96.82 | 96.84 | 96.62 |96.73
LSTM | 95.65 | 95.64| 95.41| 95.52 
BERT | 91.72 | 92.74| 90.56|91.31 
LSTM + GCN + Max Pool | 98.08 | 98.12|97.89 |98.02 
LSTM + GCN + Max Pool + Semantic Adj | 96.77 | 97.57|97.85 |97.7 
LSTM + GCN + Attn | 98.27 | 98.05| 98.42|98.22
LSTM + GCN + Attn + Semantic Adj| 98.17| 98.15| 98.06|98.11
LSTM + GAT | 98.36| 98.44| 98.12|98.29
LSTM + GAT + Semantic Adj | 98.25 | 98.29| 98.09|98.19
LSTM + GAT + 2 Attn Heads| 98.44 | 98.44| 98.34|98.39
LSTM + GAT + 2 Attn Heads + Semantic Adj | 98.02 | 98.01|97.9 |97.95

### Out of domain test set 1 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 67.5 |67.79 | 67.5|67.37
LSTM | 81.11 | 82.12| 81.11| 80.96 
BERT | 75.83| 76.62| 75.83| 75.65
LSTM + GCN + Max Pool | 85.83| 86.16| 85.83| 85.8
LSTM + GCN + Max Pool + Semantic Adj | 83.89 | 84.73| 83.89|83.79 
LSTM + GCN + Attn | 85.27 | 85.59 | 85.27 | 85.24
LSTM + GCN + Attn + Semantic Adj| 85.56 | 85.57| 85.56|85.55
LSTM + GAT |86.39| 86.44|86.38 |86.38
LSTM + GAT + Semantic Adj | 85.27| 85.31| 85.27|85.27
LSTM + GAT + 2 Attn Heads| 84.72| 85.65| 84.72|84.62
LSTM + GAT + 2 Attn Heads + Semantic Adj | 86.94 | 87.04| 86.94|86.94
SoTA | - | 90.0 | 84.0 | 87.0

### Out of domain test set 2 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 91.13 |91.28|91.13  |91.12 
LSTM | 91.53 | 91.54 | 91.53 | 91.53
BERT | 83.46 | 83.56 | 83.46| 83.45 
LSTM + GCN + Max Pool | 92.6|  92.61| 92.59|92.59 
LSTM + GCN + Max Pool + Semantic Adj | 89.73| 90.57 | 89.73|89.68 
LSTM + GCN + Self Attn | 91.26 | 91.99|91.26 |91.22
LSTM + GCN + Self Attn + Semantic Adj |92.4| 92.53 |92.39 |92.39  
LSTM + GAT | 94.2| 94.21 | 94.2| 94.19
LSTM + GAT + Semantic Adj | 92.6| 92.69 |92.59 |92.59 
LSTM + GAT + 2 Attn Heads| 89.66| 90.37 | 89.67| 89.62
LSTM + GAT + 2 Attn Heads + Semantic Adj | 92.86 | 93.06 | 92.87|92.86 

### For four classes Satire, Hoax, Propaganda and Trusted

### In domain dev set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 96.48 | 96.41 | 96.18 | 96.28 / 96.48
LSTM | 88.75| 88.67 | 88.11 | 88.35 / 88.75
BERT | 95.07| 94.81 | 94.57 | 94.68 / 95.07
LSTM + GCN + Max Pool | 96.76 | 96.61 | 96.58 | 96.59 / 96.76 
LSTM + GCN + Max Pool + Semantic Adj | |  |  |
LSTM + GCN + Attn | 97.57 | 97.25 | 97.63 | 97.43 / 97.57
LSTM + GCN + Attn + Semantic Adj| |  |  |
LSTM + GAT | 97.73| 97.9 | 97.36 | 97.62 / 97.28
LSTM + GAT + Semantic Adj ||  |  |
LSTM + GAT + 2 Attn Heads| 97.8 | 97.69 | 97.74 | 97.71 / 97.82
LSTM + GAT + 2 Attn Heads + Semantic Adj ||  |  |
SoTA | - | - | - | 91.0 |

### Out of domain test set 2 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 54.03 | 54.5 | 54.03 | 52.6 / 54.03
LSTM |55.06 | 58.88 | 55.06 | 52.5 / 55.05
BERT | 55.56 | 57.45 | 54.86 | 54.0 / 54.87
LSTM + GCN + Max Pool | 65.0 | 66.75 | 64.84 | 63.79 / 65.0
LSTM + GCN + Max Pool + Semantic Adj | |  |  |
LSTM + GCN + Attn | 67.08 | 68.6 | 67.0 | 66.42 / 67.08
LSTM + GCN + Attn + Semantic Adj| |  |  |
LSTM + GAT | 65.5| 69.45 | 65.33 | 63.83 / 65.51
LSTM + GAT + Semantic Adj | |  |  |
LSTM + GAT + 2 Attn Heads| 66.94 | 68.05 | 66.86 | 66.37 / 66.95
LSTM + GAT + 2 Attn Heads + Semantic Adj | |  |  |
SoTA | - | - | - | 65.0 |

### Document classification

### AG News (4 news categories)
Model | Acc | Test Error Rate
--- | --- | --- 
GAT | 89.61 | 10.39
GAT + 2 Attn Heads | 89.72 | 10.28
SOTA | | 5.01

### IMDB (2 sentiment categories)
Model | Acc | Test Error Rate
--- | --- | --- 
GAT |  |
GAT + 2 Attn Heads | | 
SOTA | | 4.6

### DBPedia (14 ontology categories)
Model | Acc | Test Error Rate
--- | --- | --- 
GAT | 99.13 | 0.87 
GAT + 2 Attn Heads | | 
SOTA | | 0.80


