import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data


class DataLoader:
    def __init__(self, params):
        self.params = params

        # Dictionaries for word to index and vice versa
        w2i = freezable_defaultdict(lambda: len(w2i))
        # Adding unk token
        UNK = w2i["<unk>"]

        # Read in the data and store the dicts
        if self.params.encoder >= 2:
            self.train = self.read_dataset_sentence_wise(params.train, w2i)
        else:
            self.train = list(self.read_dataset(params.train, w2i))
        print("Average train document length: {}".format(np.mean([len(x[0]) for x in self.train])))
        print("Maximum train document length: {}".format(max([len(x[0]) for x in self.train])))
        w2i = freezable_defaultdict(lambda: UNK, w2i)
        w2i.freeze()
        if self.params.encoder >= 2:
            self.dev = self.read_dataset_sentence_wise(params.dev, w2i)
        else:
            self.dev = list(self.read_dataset(params.dev, w2i))
        self.w2i = w2i
        self.i2w = dict(map(reversed, self.w2i.items()))
        self.nwords = len(w2i)
        # Treating this as a binary classification problem for now "1: Satire, 4: Trusted"
        self.ntags = 2
        if self.params.encoder >= 2:
            self.test = self.read_testset_sentence_wise(params.test, w2i)
        else:
            self.test = self.read_testset(params.test, w2i)
        # Setting pin memory and number of workers
        kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

        dataset_train = ClassificationGraphDataSet(self.train, self.params) if self.params.encoder >= 2 else \
            ClassificationDataSet(self.train, self.params)
        self.train_data_loader = torch.utils.data.DataLoader(dataset_train, batch_size=params.batch_size,
                                                             collate_fn=dataset_train.collate, shuffle=True,
                                                             **kwargs)

        dataset_dev = ClassificationGraphDataSet(self.dev, self.params) if self.params.encoder >= 2 else \
            ClassificationDataSet(self.dev, self.params)
        self.dev_data_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=params.batch_size,
                                                           collate_fn=dataset_dev.collate, shuffle=False, **kwargs)

        dataset_test = ClassificationGraphDataSet(self.test, self.params) if self.params.encoder >= 2 else \
            ClassificationDataSet(self.test, self.params)
        self.test_data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=params.batch_size,
                                                            collate_fn=dataset_test.collate, shuffle=False,
                                                            **kwargs)

    @staticmethod
    def read_dataset(filename, w2i):
        with open(filename, "r") as f:
            readCSV = csv.reader(f, delimiter=',')
            csv.field_size_limit(100000000)
            for tag, words in readCSV:
                tag = int(tag)
                if tag in [1, 4]:
                    # Adjust the tag to {0: Satire, 1: Trusted}
                    yield ([w2i[x] for x in words.lower().split(" ")], tag - 1 if tag == 1 else tag - 3)

    @staticmethod
    def read_testset(filename, w2i):
        df = pd.read_excel(filename)
        data = []
        for row in df.values:
            tag = int(row[0])
            # Tag id is reversed in this dataset
            data.append(([w2i[x] for x in row[2].lower().split(" ")], tag + 1 if tag == 0 else tag - 1))
        return data

    @staticmethod
    def read_dataset_sentence_wise(filename, w2i):
        data = []
        with open(filename, "r") as f:
            readCSV = csv.reader(f, delimiter=',')
            csv.field_size_limit(100000000)
            for tag, doc in readCSV:
                sentences = doc.split('.')
                tag = int(tag)
                if tag in [1, 4]:
                    # Adjust the tag to {0: Satire, 1: Trusted}
                    tag = tag - 1 if tag == 1 else tag - 3
                    sentences_idx = []
                    for sentence in sentences:
                        sentence = sentence.lower().strip().split(" ")
                        if len(sentence) > 1:
                            sentences_idx.append([w2i[x] for x in sentence])
                    if len(sentences_idx) > 1:
                        data.append((sentences_idx, tag))
        return data

    @staticmethod
    def read_testset_sentence_wise(filename, w2i):
        df = pd.read_excel(filename)
        data = []
        for row in df.values:
            sentences = row[2].split('.')
            tag = int(row[0])
            # Tag id is reversed in this dataset
            tag = tag + 1 if tag == 0 else tag - 1
            sentences_idx = []
            for sentence in sentences:
                sentence = sentence.lower().replace("\n", " ").strip().split(" ")
                if len(sentence) > 1:
                    sentences_idx.append([w2i[x] for x in sentence])
            if len(sentences_idx) > 1:
                data.append((sentences_idx, tag))
        return data


class ClassificationDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data, params):
        super(ClassificationDataSet, self).__init__()
        self.params = params
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.num_of_samples = len(self.sents)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx]

    def collate(self, batch):
        sents = np.array([x[0] for x in batch])
        sent_lens = np.array([min(self.params.max_sent_len, x[1]) for x in batch])
        labels = np.array([x[2] for x in batch])

        # List of indices according to decreasing order of sentence lengths
        sorted_input_seq_len = np.flipud(np.argsort(sent_lens))
        # Sorting the elements od the batch in decreasing length order
        input_lens = sent_lens[sorted_input_seq_len]
        sents = sents[sorted_input_seq_len]
        labels = labels[sorted_input_seq_len]

        # Creating padded sentences
        sent_max_len = min(input_lens[0], self.params.max_sent_len)
        padded_sents = np.zeros((len(batch), sent_max_len))
        for i, sent in enumerate(sents):
            padded_sents[i, :len(sent)] = sent[:sent_max_len]

        return padded_sents, input_lens, labels


class ClassificationGraphDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data, params):
        super(ClassificationGraphDataSet, self).__init__()
        self.params = params
        # data is a list of tuples (sent, label)
        self.sents = [x[0] for x in data]
        self.labels = [x[1] for x in data]
        self.num_of_samples = len(self.sents)

    def __len__(self):
        return self.num_of_samples

    def __getitem__(self, idx):
        return self.sents[idx], len(self.sents[idx]), self.labels[idx]

    def collate(self, batch):
        sents = np.array([x[0] for x in batch])
        doc_lens = np.array([x[1] for x in batch])
        labels = np.array([x[2] for x in batch])
        # Sort sentences within each document by length
        documents = []

        for doc in sents:
            curr_lens = np.array([min(self.params.max_sent_len, len(x)) for x in doc])
            curr_sents = np.array(doc)
            sorted_input_seq_len = np.flipud(np.argsort(curr_lens))
            curr_sents = curr_sents[sorted_input_seq_len]
            curr_lens = curr_lens[sorted_input_seq_len]

            padded_sents = np.zeros((len(curr_sents), curr_lens[0]))
            for i, sen in enumerate(curr_sents):
                padded_sents[i, :len(sen)] = sen[:curr_lens[0]]
            documents.append((padded_sents, curr_lens))

        return documents, doc_lens, labels


class freezable_defaultdict(dict):
    def __init__(self, default_factory, *args, **kwargs):
        self.frozen = False
        self.default_factory = default_factory
        super(freezable_defaultdict, self).__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.frozen:
            return self.default_factory()
        else:
            self[key] = value = self.default_factory()
            return value

    def freeze(self):
        self.frozen = True
