# From: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import time
import torchvision.transforms as T
import wandb
import re
import argparse

# Get cpu, gpu device for training.
# mps does not (yet) support nn.EmbeddingBag.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CsvTextDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        text = self.data_frame.loc[idx, "article"]
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            text = self.transform(text)

        return text, label

class SimpleTokenizer:
    def __call__(self, text):
        # Add a space between punctuation and words
        text = re.sub(r'([.,:;!?()])', r' \1 ', text)
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize by splitting on whitespace
        return text.split()

class Vocab:
    def __init__(self, oov_token, pad_token):
        self.idx2str = []
        self.str2idx = {}
        self.oov_index = 0
        self.add_tokens([oov_token, pad_token])
        self.oov_idx = self[oov_token]
        self.pad_idx = self[pad_token]

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.str2idx:
                self.str2idx[token] = len(self.idx2str)
                self.idx2str.append(token)

    def __len__(self):
        return len(self.str2idx)

    def __getitem__(self, token):
        return self.str2idx.get(token, self.oov_index)

class CorpusInfo():
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.oov_token = '<OOV>' # out-of-vocabulary token
        self.pad_token = '<PAD>' # padding token
        
        self.vocab = Vocab(self.oov_token, self.pad_token)
        for text, _ in dataset:
            self.vocab.add_tokens(tokenizer(text))
        
        self.oov_idx = self.vocab[self.oov_token]
        self.pad_idx = self.vocab[self.pad_token]
        
        self.vocab_size = len(self.vocab)
        self.num_labels = len(set([label for (text, label) in dataset]))

class TextTransform():
    def __init__(self, tokenizer, vocab):
        self.tokenizer = tokenizer
        self.vocab = vocab

    def tokenize_and_numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.vocab[token] for token in tokens]

    def __call__(self, text):
        return self.tokenize_and_numericalize(text)
    
class ToIntTensor():
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.int64)

def length_histogram(train_data, pad_idx):
    raise NotImplementedError("TODO: implement length_histogram")

def get_data(args):    
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=None,
    )
    tokenizer = SimpleTokenizer()
    corpus_info = CorpusInfo(train_data, tokenizer)
    if args.max_len < 1:
        print("No padding or truncation will be applied.")
        transform_txt = T.Compose([
            TextTransform(corpus_info.tokenizer, corpus_info.vocab),
            ToIntTensor(),
        ])
    else:
        print("Padding and truncation will be applied.")
        raise NotImplementedError("TODO: Implement two transforms: TruncateToMaxLen and PadToMaxLen")
        transform_txt = T.Compose([
            TextTransform(corpus_info.tokenizer, corpus_info.vocab),
            TruncateToMaxLen(args.max_len),
            PadToMaxLen(args.max_len, corpus_info.pad_idx),
            ToIntTensor(),
        ])
        
    train_data = CsvTextDataset(
        csv_file='./data/txt_train.csv',
        transform=transform_txt,
    )
    val_data = CsvTextDataset(
        csv_file='./data/txt_val.csv',
        transform=transform_txt,
    )
    test_data = CsvTextDataset(
        csv_file='./data/txt_test.csv',
        transform=transform_txt,
    )

    if args.length_histogram:
        length_histogram(train_data, corpus_info.pad_idx)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, N]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        print(f"Sample X: {X}")
        print(f"Sample y: {y}")
        break
    
    return corpus_info, train_dataloader, val_dataloader, test_dataloader

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)


def train_one_epoch(dataloader, model, criterion, optimizer, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 5
    start_time = time.time()

    for idx, (text, label) in enumerate(dataloader):
        text, label = text.to(device), label.to(device)
        optimizer.zero_grad()
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0
            start_time = time.time()


def evaluate(dataloader, model, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (text, label) in enumerate(dataloader):
            text, label = text.to(device), label.to(device)
            predicted_label = model(text)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count

def main(args):
    torch.manual_seed(10999)
    
    if args.use_wandb:
        raise NotImplementedError("TODO: implement wandb logging.")
    else:
        wandb.init(mode='disabled')
        
    corpus_info, train_dataloader, val_dataloader, test_dataloader = get_data(args)

    if args.model == 'simple':
        model = TextClassificationModel(corpus_info.vocab_size, args.embed_dim, corpus_info.num_labels).to(device)
    elif args.model == 'lstm':
        raise NotImplementedError("TODO: implement LSTM model.")
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        raise NotImplementedError("TODO: implement Adam optimizer.")
    else:
        raise ValueError(f"Unknown optimizer type: {args.optimizer}")
        
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

    total_accu = None    
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_one_epoch(train_dataloader, model, criterion, optimizer, epoch)
        accu_val = evaluate(val_dataloader, model, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        print("-" * 59)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | "
            "valid accuracy {:8.3f} ".format(
                epoch, time.time() - epoch_start_time, accu_val
            )
        )
        print("-" * 59)

    print("Checking the results of test dataset.")
    accu_test = evaluate(test_dataloader, model, criterion)
    print("test accuracy {:8.3f}".format(accu_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Text Classifier Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=5.0, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=1024, help='Maximum text input length')
    parser.add_argument('--model', type=str, choices=['simple', 'lstm'], default='simple', help='Model type to use')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer type to use')
    parser.add_argument('--length_histogram', action='store_true', default=False, help='Plot length histogram')
    
    args = parser.parse_args()
    main(args)