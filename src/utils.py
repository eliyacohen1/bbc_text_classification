from typing import Tuple
from dataset import DatasetForBert
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from config import config

tokenizer_config = config["tokenizer_config"]
config_labels = config["labels"]

np.random.seed(0)


def evaluation():
    pass


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def generate_datasets(df: pd.DataFrame, val_size: float, test_size: float, batch_size):
    train, validation = train_test_split(df, test_size=(val_size + test_size))
    validation, test = train_test_split(validation, test_size=test_size / (val_size + test_size))
    print(len(train), len(validation), len(test))
    test_dataloader, train_dataloader, val_dataloader = create_dataloader(batch_size, test, train, validation)
    return train_dataloader, val_dataloader, test_dataloader


def create_dataloader(batch_size, test, train, validation):
    print("Start to create DataLoader")
    train_data = DatasetForBert(train.Text, train.Category, config_labels, tokenizer_config)
    val_data = DatasetForBert(validation.Text, validation.Category, config_labels, tokenizer_config)
    test_data = DatasetForBert(test.Text, test.Category, config_labels, tokenizer_config)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=test.shape[0])

    return test_dataloader, train_dataloader, val_dataloader


def train_model(model, train_dataloader, val_dataloader, learning_rate=0.01, epochs=5):
    print("Start Train")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        print("epoch", epoch_num)
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataloader): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataloader): .3f} \
                | Val Loss: {total_loss_val / len(val_dataloader): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataloader): .3f}')
