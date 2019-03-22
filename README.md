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

To train a LSTM model, run the following command,
```
python main.py --batch_size 1024 --config lstm --encoder 0 --mode 0
```

To train a CNN model, run the following command,
```
python main.py --batch_size 1024 --config cnn --encoder 1 --mode 0
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

## Baseline Results

Model | In domain test set accuracy | Out of domain test set accuracy
--- | --- | ---
CNN | 92.3 | 67.5
LSTM | 91.4 | 81.4