import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchtext.datasets as dts
import torchtext.data as dt
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm


class CNNText(nn.Module):
    """
    Convolutional Neural Network for Sentence Classification with fixed sentence length
    """
    def __init__(self, vocab, targetset, sentence_len, pad_idx, word_embedding_dim, kernel_sizes=[3, 4, 5], num_filters=100, dropout_keep_prob=0.5):
        super(CNNText, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.targetset = targetset
        self.targetset_size = len(targetset)
        self.sentence_len = sentence_len
        self.pad_idx = pad_idx

        self.word_embedding_dim = word_embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob

        self._build_model()

    def _get_maxpooling_over_time_kernel_size(self, conv_kernel_size, padding=0, dilation=1, stride=1):
        return math.floor((self.sentence_len + 2. * padding - dilation * (conv_kernel_size - 1.) - 1.) / stride + 1.)

    def _build_model(self):
        self.word_embeds = nn.Embedding(self.vocab_size, self.word_embedding_dim, padding_idx=self.pad_idx)
        conv_blocks = []
        for kernel_size in self.kernel_sizes:
            conv1d = nn.Conv1d(in_channels=self.word_embedding_dim, out_channels=self.num_filters, kernel_size=kernel_size, stride=1)

            maxpool_kernel_size = self._get_maxpooling_over_time_kernel_size(conv_kernel_size=kernel_size)
            component = nn.Sequential(
                conv1d,
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=maxpool_kernel_size)
            )
            conv_blocks.append(component)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dropout = nn.Dropout(p=self.dropout_keep_prob)
        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), self.targetset_size)

    def forward(self, sents):
        word_embedding = self.word_embeds(sents)
        word_embedding_for_conv_block = word_embedding.permute(0, 2, 1).contiguous()

        conv_outputs = [conv_block(word_embedding_for_conv_block) for conv_block in self.conv_blocks]
        out = torch.cat(conv_outputs, 2)
        sent_feats = out.view(out.size(0), -1)
        sent_feats = self.dropout(sent_feats)
        class_scores = self.fc(sent_feats)
        return class_scores


class DynamicCNNText(nn.Module):
    """
    Convolutional Neural Network with dynamic sentence length in batch setting
    """
    def __init__(self, vocab, targetset, pad_idx, word_embedding_dim, kernel_sizes=[3, 4, 5], num_filters=100, dropout_keep_prob=0.5):
        super(DynamicCNNText, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.targetset = targetset
        self.targetset_size = len(targetset)
        self.pad_idx = pad_idx

        self.word_embedding_dim = word_embedding_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.dropout_keep_prob = dropout_keep_prob

        self._build_model()

    def _relu_and_maxpool(self, inputs):
        inputs = [F.relu(i) for i in inputs]
        inputs = [F.max_pool1d(i, i.size(2)) for i in inputs]
        return inputs

    def _build_model(self):
        self.word_embeds = nn.Embedding(self.vocab_size, self.word_embedding_dim, padding_idx=self.pad_idx)
        conv_blocks = []
        for kernel_size in self.kernel_sizes:
            conv1d = nn.Conv1d(in_channels=self.word_embedding_dim, out_channels=self.num_filters, kernel_size=kernel_size, stride=1)
            conv_blocks.append(conv1d)

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dropout = nn.Dropout(p=self.dropout_keep_prob)
        self.fc = nn.Linear(self.num_filters * len(self.kernel_sizes), self.targetset_size)

    def forward(self, sents):
        word_embedding = self.word_embeds(sents)
        word_embedding_for_conv_block = word_embedding.permute(0, 2, 1).contiguous()

        conv_outputs = [conv_block(word_embedding_for_conv_block) for conv_block in self.conv_blocks]
        conv_outputs = self._relu_and_maxpool(conv_outputs)
        out = torch.cat(conv_outputs, 2)
        sent_feats = out.view(out.size(0), -1)
        sent_feats = self.dropout(sent_feats)
        class_scores = self.fc(sent_feats)
        return class_scores


def evaluation_with_torchtext(model, data_iter, nb_examples):
    accuracy = 0.0
    model.eval()
    total_preds_classes = np.array([])
    total_targets_classes = np.array([])
    for batch_idx, batch in enumerate(data_iter):
        # get batch data
        batch_inputs = batch.text
        batch_targets = batch.label
        total_targets_classes = np.append(total_targets_classes, batch_targets.cpu().numpy())

        # forward
        batch_pred_scores = model(batch_inputs)

        #
        highest_scores, pred_classes = batch_pred_scores.max(1)
        pred_classes = pred_classes.view(-1)
        accuracy += pred_classes.eq(batch_targets).sum().item()
        total_preds_classes = np.append(total_preds_classes, pred_classes.cpu().numpy())

    accuracy /= nb_examples

    precision, recall, f1, support = metrics.precision_recall_fscore_support(total_targets_classes, total_preds_classes,
                                                                             average="micro")

    return accuracy, precision, recall, f1


def train_with_torchtext(model, optimizer, datasets, batch_size=32, epochs=50, use_gpu=False,
                         final_test=True, verbose=0):
    loss_func = nn.CrossEntropyLoss()
    # get train, validation, test data parts
    # make train, validation, test SST iterators
    train_iter, valid_iter, test_iter = dt.BucketIterator.splits(datasets,
                                                                 batch_sizes=(batch_size, batch_size, batch_size),
                                                                 shuffle=True, device="cuda" if use_gpu else None)
    # logging
    if verbose != 0:
        print("Number of training examples: {}".format(len(datasets[0])))

    with tqdm(desc="Training", total=epochs) as pbar:
        for epoch_idx in range(epochs):
            # shuffle training data
            train_iter.shuffle
            model.train()
            # training log info
            epoch_loss = 0.0

            # mini-batch training
            for batch_idx, batch in enumerate(train_iter):
                # get batch data
                batch_inputs = batch.text
                batch_targets = batch.label

                # debug
                # print(batch_inputs.size())
                # print(batch_targets.size())

                # forward
                batch_pred_scores = model(batch_inputs)
                # loss
                batch_loss = loss_func(batch_pred_scores, batch_targets)

                # backpropagation
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                # logging
                if verbose == 2:
                    print("Batch {}: {:.2f}".format(batch_idx + 1, batch_loss.item()))

                # epoch loss, accuracy
                epoch_loss += batch_loss.item() * batch.batch_size  # batch_loss reduced in average

            epoch_loss /= len(datasets[0])
            valid_accuracy, valid_precision, valid_recall, valid_f1 = evaluation_with_torchtext(model, valid_iter,
                                                                                                len(datasets[1]))

            pbar.update()
            pbar.set_postfix({
                "Traing Loss": "{:.3f}".format(epoch_loss),
                "Validation Accuracy": "{:.3f}".format(valid_accuracy),
                "Validation F1": "{:.3f}".format(valid_f1)
            })

            if verbose != 0:
                print("Epoch {}: Train Loss:{:.3f} | Valid accuracy: {:.3f} | Valid F1: {:.3f}".format(epoch_idx + 1,
                                                                                                       epoch_loss,
                                                                                                       valid_accuracy,
                                                                                                       valid_f1), end="\r")
    if final_test:
        test_accuracy, test_precision, test_recall, test_f1 = evaluation_with_torchtext(model, test_iter,
                                                                                        len(datasets[2]))
        print("\nTest Accuracy: {:.3f} | Test F1: {:.3f}".format(test_accuracy, test_f1))
    return model


if __name__ == "__main__":
    # set up fields of torchtext dataset
    TEXT = dt.Field(lower=True, include_lengths=False, batch_first=True)
    LABEL = dt.Field(sequential=False)

    # load SST dataset
    sst_dataset = dts.SST.splits(TEXT, LABEL)

    # build the vocabulary
    TEXT.build_vocab(sst_dataset[0])
    LABEL.build_vocab(sst_dataset[0])

    # vocabulary, target set and inverted of them used to debug
    word_to_idx = TEXT.vocab.stoi
    target_to_idx = LABEL.vocab.stoi
    idx_to_word = TEXT.vocab.itos
    idx_to_target = LABEL.vocab.itos
    pad_sym = "<pad>"
    print("Index of PAD symbol: {}".format(word_to_idx[pad_sym]))
    total_data_examples = sum([len(data_part) for data_part in sst_dataset])
    print("Total examples: {}".format(total_data_examples))

    # Network setting
    batch_size = 50
    word_embedding_dim = 128
    learning_rate = 0.1
    l2_weight = 0.
    dropout_keep_prob = 0.5
    kernel_sizes = [3, 4, 5]
    num_filters = 100
    use_gpu = True
    run_testing = True
    nb_epochs = 200


    # model
    sst_cnn = DynamicCNNText(word_to_idx, target_to_idx, pad_idx=word_to_idx[pad_sym],
                             word_embedding_dim=word_embedding_dim,
                             kernel_sizes=kernel_sizes, num_filters=num_filters,
                             dropout_keep_prob=dropout_keep_prob)
    if use_gpu:
        sst_cnn.cuda()

    # optimizer
    optimizer = torch.optim.Adadelta(sst_cnn.parameters(), lr=learning_rate, weight_decay=l2_weight)

    sst_cnn = train_with_torchtext(sst_cnn, optimizer, sst_dataset, batch_size, epochs=nb_epochs,
                                   use_gpu=use_gpu, final_test=run_testing, verbose=0)

