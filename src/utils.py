from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.models import TextClassificationInterface
from src.models.tokenizer_model import TokenizerModelInterface
from .dataset.dataset_interface import DatasetInterface

np.random.seed(0)


def load_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def split_data(df: pd.DataFrame, val_size: float, test_size: float):
    train, validation = train_test_split(df, test_size=(val_size + test_size))
    validation, test = train_test_split(validation, test_size=test_size / (val_size + test_size))
    return train, validation, test


def create_datasets(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, dataset_obj,
                    tokenizer: TokenizerModelInterface, labels_map: dict):
    train_data = dataset_obj(train.Text, train.Category, labels_map, tokenizer)
    val_data = dataset_obj(validation.Text, validation.Category, labels_map, tokenizer)
    test_data = dataset_obj(test.Text, test.Category, labels_map, tokenizer)
    return train_data, val_data, test_data


def create_dataloader(dataset_train: DatasetInterface = None, dataset_validation: DatasetInterface = None,
                      dataset_test: DatasetInterface = None, batch_size: int = 16):
    dataloders = []
    if dataset_train:
        dataloders.append(DataLoader(dataset_train, batch_size=batch_size, shuffle=True))
    if dataset_validation:
        dataloders.append(DataLoader(dataset_validation, batch_size=batch_size, shuffle=True))
    if dataset_test:
        dataloders.append(DataLoader(dataset_test, batch_size=batch_size, shuffle=True))

    return dataloders


def train_model(model, train_dataloader, val_dataloader,
                learning_rate=0.01, epochs=1, optimizer=torch.optim.Adam, weight_decay: float = 0.0,
                device: str = "cpu"):
    train_length, val_length = len(train_dataloader.dataset), len(val_dataloader.dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if device != "cpu":
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        counter = 0
        for train_input, train_label in train_dataloader:
            counter += 1
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            if counter % 10 == 0:
                print(f"Accuracy at epoch {epoch_num} and batch {counter}: {acc} / {train_label.shape[0]}")
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
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / train_length: .3f} \
                    | Train Accuracy: {total_acc_train / train_length} \
                    | Val Loss: {total_loss_val / val_length: .3f} \
                    | Val Accuracy: {total_acc_val / val_length: .3f}')

    return model


def evaluate(model, test_dataloader, labels_map: dict, device="cpu"):
    if device != "cpu":
        model = model.cuda()

    y_pred = []
    y_test = []
    with torch.no_grad():

        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            y_pred += output.argmax(dim=1).tolist()
            y_test += test_label.tolist()

    labels = sorted(list(labels_map.items()), key=lambda x: x[1])
    labels = [label[0] for label in labels]
    results = classification_report(y_test, y_pred, target_names=labels)
    return results


def predict(text: str, model: TextClassificationInterface, tokenizer: TokenizerModelInterface, idx_to_labels, top_k=1,
            device="cpu"):
    model.eval()
    tokens = tokenizer.tokenize_text(text)
    mask = tokens['attention_mask'].to(device)
    input_id = tokens['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    results = torch.topk(output, top_k, dim=1).indices[0]
    return [idx_to_labels[result.item()] for result in results]
