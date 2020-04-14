from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split
import csv
from tqdm import tqdm
import torch


def get_dataloader(tokenizer, normal_file, sarcasm_file, batch_size, max_len, portion=0.75):
    labels = []
    sentences = []
    print("loading data ...")
    with open(normal_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first row
        for row in reader:
            labels.append(int(row[0]))
            sentences.append([row[2], row[1]])
    
    with open(sarcasm_file) as csv_file:
        reader =csv.reader(csv_file)
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