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

### In domain dev set accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 91.93| 91.92| 91.47|91.67
LSTM | 93.22 | 93.17| 92.88| 93.02 
BERT |  | | | 
LSTM + GCN + Max Pool |  | | | 
LSTM + GCN + Max Pool + Semantic Adj |  | | | 
LSTM + GCN + Attn |  | | |
LSTM + GCN + Attn + Semantic Adj|| | |
LSTM + GAT | | | |
LSTM + GAT + Semantic Adj |  | | |
LSTM + GAT + 2 Attn Heads|  | | |
LSTM + GAT + 2 Attn Heads + Semantic Adj |  | | |

### Out of domain test set 1 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 64.44 |64.47 | 64.44|64.43
LSTM | 68.89 | 69.04| 68.89| 68.83 
BERT | | | | 
LSTM + GCN + Max Pool | | | | 
LSTM + GCN + Max Pool + Semantic Adj |  | | | 
LSTM + GCN + Attn |  | | |
LSTM + GCN + Attn + Semantic Adj|  | | |
LSTM + GAT || | |
LSTM + GAT + Semantic Adj |  | | |
LSTM + GAT + 2 Attn Heads|  | | |
LSTM + GAT + 2 Attn Heads + Semantic Adj |  | | |

### Out of domain test set 2 accuracy
Model | Acc | Prec | Recall | F1
--- | --- | --- | --- | ---
CNN | 82.27 |82.27|82.27  |82.27 
LSTM | 84.07 | 84.18 | 84.07 | 84.05
BERT |  |  | | 
LSTM + GCN + Max Pool | |  | | 
LSTM + GCN + Max Pool + Semantic Adj | |  | | 
LSTM + GCN + Self Attn ||  | | 
LSTM + GCN + Self Attn + Semantic Adj ||  | |  
LSTM + GAT | |  | | 
LSTM + GAT + Semantic Adj | |  | | 
LSTM + GAT + 2 Attn Heads| |  | | 
LSTM + GAT + 2 Attn Heads + Semantic Adj |  |  | | 
