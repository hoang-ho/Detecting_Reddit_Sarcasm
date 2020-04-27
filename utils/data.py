import csv
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import pandas as pd


def get_dataloader_ALBERT(tokenizer, data_file, batch_size, max_len, portion=0.75):
    labels = []
    sentences = []
    print("loading data ...")
    with open(data_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first row
        for row in reader:
            labels.append(int(row[0]))
            sentences.append([row[2], row[1]])

    input_ids = []
    type_ids = []
    attention_mask = []
    print("Encoding sentences ...")
    for sent in tqdm(sentences):
        input_ = tokenizer.encode_plus(sent[0], sent[1], max_length=max_len, 
                                       pad_to_max_length=True, return_token_type_ids=True, 
                                       return_attention_mask=True, return_tensors = 'pt')
        input_ids.append(input_["input_ids"])
        type_ids.append(input_["token_type_ids"])
        attention_mask.append(input_["attention_mask"])
    
    print("Get dataloader ...")
    input_ids = torch.cat(input_ids, dim=0)
    print(input_ids.shape)
    attention_mask = torch.cat(attention_mask, dim=0)
    type_ids = torch.cat(type_ids, dim=0)
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(input_ids, labels, attention_mask, type_ids)
    print(len(dataset))
    train_size = int(portion * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size, val_size)
    train_data, validation_data = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)
    print(len(train_dataloader), len(validation_dataloader))
    
    return train_dataloader, validation_dataloader

def get_dataloader_LSTM(tokenizer, data_file, batch_size, max_len, portion=0.75):
    labels = []
    parent = []
    reply = []
    print("loading data ...")
    with open(data_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first row
        for row in reader:
            labels.append(int(row[0]))
            parent.append(row[2])
            reply.append(row[1])

    p_input_ids = []
    p_attention_mask = []
    r_input_ids = []
    r_attention_mask = []
    print("Encoding sentences ...")
    for sent in tqdm(parent):
        input_ = tokenizer.encode_plus(sent, max_length=max_len, pad_to_max_length=True, 
                                        return_attention_mask=True, return_tensors = 'pt')
        p_input_ids.append(input_["input_ids"])
        p_attention_mask.append(input_["attention_mask"])
    
    for sent in tqdm(reply):
        input_ = tokenizer.encode_plus(sent, max_length=max_len, pad_to_max_length=True, 
                                        return_attention_mask=True, return_tensors = 'pt')
        r_input_ids.append(input_["input_ids"])
        r_attention_mask.append(input_["attention_mask"])

    
    print("Get dataloader ...")
    p_input_ids = torch.cat(p_input_ids, dim=0)
    print(p_input_ids.shape)
    p_attention_mask = torch.cat(p_attention_mask, dim=0)
    labels = torch.tensor(labels)

    r_input_ids = torch.cat(r_input_ids, dim=0)
    print(r_input_ids.shape)
    r_attention_mask = torch.cat(r_attention_mask, dim=0)
    
    dataset = TensorDataset(p_input_ids, r_input_ids, p_attention_mask, r_attention_mask, labels)
    print(len(dataset))
    train_size = int(portion * len(dataset))
    val_size = len(dataset) - train_size
    print(train_size, val_size)
    train_data, validation_data = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    validation_dataloader = DataLoader(validation_data, sampler=SequentialSampler(validation_data), batch_size=batch_size)
    print(len(train_dataloader), len(validation_dataloader))
    
    return train_dataloader, validation_dataloader