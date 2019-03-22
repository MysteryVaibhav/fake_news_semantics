import torch
from model import Classify
from tqdm import tqdm


class Evaluator:
    def __init__(self, params, utils, data_loader):
        self.params = params
        self.utils = utils
        self.data_loader = data_loader

    def evaluate(self):
        model = Classify(self.params, vocab_size=len(self.data_loader.w2i),
                         ntags=self.data_loader.ntags, pte=None)
        if torch.cuda.is_available():
            model = model.cuda()
        # Load the model weights
        model.load_state_dict(torch.load("models/" + self.params.model_file, map_location=lambda storage, loc: storage))

        hits = 0
        total = 0
        model.eval()
        for sents, lens, labels in tqdm(self.data_loader.test_data_loader):
            x_batch = self.utils.to_tensor(sents)
            y_batch = self.utils.to_tensor(labels)
            logits = model(x_batch, lens)
            hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            total += len(sents)

        accuracy = hits / total
        print("Accuracy on the OOD test set: {}".format(accuracy))

