import argparse
import torch
import csv
import sys, os
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, BertForSequenceClassification
from tqdm import tqdm


class ComputeSimilarity:
    def __init__(self, args):
        self.model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=args.num_labels)
        self.tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        self.max_sentence_length = args.max_sentence_length
        print('Loaded model with fine-tuned weights')

        # Set to eval mode
        self.model.cuda()
        self.model.eval()

    def get_similarity(self, tokens_a, tokens_b):
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        token_segments = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        token_vector = self.tokenizer.convert_tokens_to_ids(tokens)
        token_vector_mask = [1] * len(token_vector)

        with torch.no_grad():
            input_ids = torch.LongTensor(token_vector).unsqueeze(0)
            segment_ids = torch.LongTensor(token_segments).unsqueeze(0)
            input_mask = torch.LongTensor(token_vector_mask).unsqueeze(0)
            out = self.model(input_ids, segment_ids, input_mask, labels=None)

        return out.item()

    def get_similarity_batched(self, tokens_a, tokenized_list):
        input_ids, segment_ids, input_mask = [], [], []
        max_token_len = 0

        for tokens_b in tokenized_list:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
            token_vector = self.tokenizer.convert_tokens_to_ids(tokens)

            input_ids.append(token_vector)
            segment_ids.append([0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1))
            input_mask.append([1] * len(token_vector))
            max_token_len = max(max_token_len, len(token_vector))

        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [0]*(max_token_len-len(input_ids[i]))
            segment_ids[i] = segment_ids[i] + [0] * (max_token_len - len(segment_ids[i]))
            input_mask[i] = input_mask[i] + [0] * (max_token_len - len(input_mask[i]))

        with torch.no_grad():
            input_ids = torch.LongTensor(input_ids).cuda()
            segment_ids = torch.LongTensor(segment_ids).cuda()
            input_mask = torch.LongTensor(input_mask).cuda()
            out = self.model(input_ids, segment_ids, input_mask, labels=None)

        return out[:, 0].cpu().data.numpy()


    def get_similarity_scores(self, sentences):
        # Compares sentences with themselves as well
        # ie. non-zero diagonal score

        num_sentences = len(sentences)
        adj_matrix = []

        tokenized_sentences = []
        for sentence in sentences:
            tokenized_sentences.append(self.tokenizer.tokenize(sentence)[:self.max_sentence_length])

        for i in tqdm(range(num_sentences)):
            adj_matrix.append(self.get_similarity_batched(tokenized_sentences[i], tokenized_sentences))

        return np.array(adj_matrix)


def doc_to_sentences(doc):
    return [x.lower() for x in doc.strip().split(".")]


def read_csv_file(filename):
    examples = []
    with open(filename, "r") as f:
        readCSV = csv.reader(f, delimiter=',')
        csv.field_size_limit(100000000)
        for tag, words in readCSV:
            tag = int(tag)
            if tag in [1, 4]:
                examples.append(words.lower())
    return examples


def read_xlsx_file(filename):
    df = pd.read_excel(filename)
    data = []
    for row in df.values:
        tag = int(row[0])
        data.append(row[2].lower())
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_type", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--num_labels", default=1, type=int)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_sentence_length", default=125, type=int)
    parser.add_argument("--part_size", default=1000, type=int)
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()

    similarity_helper = ComputeSimilarity(args)

    if args.mode == 'train':
        documents = read_csv_file('../data/newsfiles/fulltrain.csv')
    elif args.mode == 'dev':
        documents = read_csv_file('../data/newsfiles/balancedtest.csv')
    elif args.mode == 'test':
        documents = read_xlsx_file('../data/newsfiles/test.xlsx')
    else:
        sys.exit(0)

    save_directory = '%s/%s-parts/' % (args.output_dir, args.mode)
    os.makedirs(save_directory) if not os.path.exists(save_directory) else None

    adj_matrix_list = []
    part_ctr = 0
    for i in tqdm(range(len(documents))):
        doc = documents[i]
        adj_matrix_list.append(similarity_helper.get_similarity_scores(doc_to_sentences(doc)))

        if i>0 and i % args.part_size == 0:
            # Flush contents of adj_matrix_list
            adj_matrix_list = np.array(adj_matrix_list)
            np.save("%s/%s-adj_matrix-%d.npy" % (save_directory, args.mode, part_ctr), adj_matrix_list)
            part_ctr += 1
            adj_matrix_list = []

    if len(adj_matrix_list) > 0:
        np.save("%s/%s-adj_matrix-%d.npy" % (save_directory, args.mode, part_ctr), adj_matrix_list)
