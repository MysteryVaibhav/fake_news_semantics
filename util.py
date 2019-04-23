from timeit import default_timer as timer
import numpy as np
from tqdm import tqdm
from model import Classify
import torch
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Utils:
    def __init__(self, params, dl):
        self.params = params
        self.data_loader = dl

    @staticmethod
    def to_tensor(arr):
        # list -> Tensor (on GPU if possible)
        if torch.cuda.is_available():
            tensor = torch.tensor(arr).type(torch.cuda.LongTensor)
        else:
            tensor = torch.tensor(arr).type(torch.LongTensor)
        return tensor

    def get_dev_loss_and_acc(self, model, loss_fn):
        losses = []
        hits = 0
        total = 0
        model.eval()
        for sents, lens, labels, adjs in self.data_loader.dev_data_loader:
            y_batch = self.to_tensor(labels)
            if self.params.encoder >= 2:
                # This is currently unbatched
                logits = self.get_gcn_logits(model, sents, adjs)
            else:
                x_batch = self.to_tensor(sents)
                logits = model(x_batch, lens)

            loss = loss_fn(logits, y_batch)
            hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
            total += len(sents)
            losses.append(loss.item())

        return np.asscalar(np.mean(losses)), hits / total

    def get_gcn_logits(self, model, docs, adjs, actual_sentences=None):
        logits = []
        for i, (sents, sent_lens) in enumerate(docs):
            x_batch = self.to_tensor(sents)
            if actual_sentences is not None:
                logit = model(x_batch, sent_lens, adjs[i] if adjs is not None else None, actual_sentences[i])
            else:
                logit = model(x_batch, sent_lens, adjs[i] if adjs is not None else None)
            logits.append(logit)
        return torch.stack(logits)

    def train(self, pretrained_emb, save_plots_as):
        model = Classify(self.params, vocab_size=len(self.data_loader.w2i),
                         ntags=self.data_loader.ntags, pte=pretrained_emb)
        loss_fn = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr, weight_decay=self.params.weight_decay)

        # Variables for plotting
        train_losses = []
        dev_losses = []
        train_accs = []
        dev_accs = []
        s_t = timer()
        prev_best = 0
        patience = 0

        # Start the training loop
        for epoch in range(1, self.params.max_epochs + 1):
            model.train()
            train_loss = 0
            hits = 0
            total = 0
            for sents, lens, labels, adjs in tqdm(self.data_loader.train_data_loader):
                y_batch = self.to_tensor(labels)

                if self.params.encoder >= 2:
                    # This is currently unbatched
                    logits = self.get_gcn_logits(model, sents, adjs)
                else:
                    # Converting data to tensors
                    x_batch = self.to_tensor(sents)

                    # Forward pass
                    logits = model(x_batch, lens)

                loss = loss_fn(logits, y_batch)

                # Book keeping
                train_loss += loss.item()
                hits += torch.sum(torch.argmax(logits, dim=1) == y_batch).item()
                # One can alternatively do this accuracy computation on cpu by,
                # moving the logits to cpu: logits.data.cpu().numpy(), and then using numpy argmax.
                # However, we should always avoid moving tensors between devices if possible for faster computation
                total += len(sents)

                # Back-prop
                optimizer.zero_grad()  # Reset the gradients
                loss.backward()  # Back propagate the gradients
                optimizer.step()  # Update the network

            # Compute loss and acc for dev set
            dev_loss, dev_acc = self.get_dev_loss_and_acc(model, loss_fn)
            train_losses.append(train_loss / len(self.data_loader.train_data_loader))
            dev_losses.append(dev_loss)
            train_accs.append(hits / total)
            dev_accs.append(dev_acc)
            tqdm.write("Epoch: {}, Train loss: {}, Train acc: {}, Dev loss: {}, Dev acc: {}".format(
                        epoch, train_loss, hits / total, dev_loss, dev_acc))
            if dev_acc < prev_best:
                patience += 1
                if patience == 3:
                    # Reduce the learning rate by a factor of 2 if dev acc doesn't increase for 3 epochs
                    # Learning rate annealing
                    optim_state = optimizer.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2
                    optimizer.load_state_dict(optim_state)
                    tqdm.write('Dev accuracy did not increase, reducing the learning rate by 2 !!!')
                    patience = 0
            else:
                prev_best = dev_acc
                # Save the model
                torch.save(model.state_dict(), "models/model_{}.t7".format(save_plots_as))

        # Acc vs time plot
        fig = plt.figure()
        plt.plot(range(1, self.params.max_epochs + 1), train_accs, color='b', label='train')
        plt.plot(range(1, self.params.max_epochs + 1), dev_accs, color='r', label='dev')
        plt.ylabel('accuracy')
        plt.xlabel('epochs')
        plt.legend()
        plt.xticks(np.arange(1, self.params.max_epochs + 1, step=4))
        fig.savefig('data/' + '{}_accuracy.png'.format(save_plots_as))

        return timer() - s_t

    def get_pre_trained_embeddings(self):
        print("Reading pre-trained embeddings...")
        embeddings = np.random.uniform(-0.25, 0.25, (len(self.data_loader.w2i), self.params.emb_dim))
        count = 0
        with open(self.params.pte, 'r', encoding='utf-8') as f:
            ignore_first_row = True
            for row in f.readlines():
                if ignore_first_row:
                    ignore_first_row = False
                    continue
                split_row = row.split(" ")
                vec = np.array(split_row[1:-1]).astype(np.float)
                if split_row[0] in self.data_loader.w2i and len(vec) == self.params.emb_dim:
                    embeddings[self.data_loader.w2i[split_row[0]]] = vec
                    count += 1
        print("Successfully loaded {} embeddings out of {}".format(count, len(self.data_loader.w2i)))
        return np.array(embeddings)
