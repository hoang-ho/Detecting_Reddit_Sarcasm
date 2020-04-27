import torch
import csv
import h5py
from tqdm import tqdm
from transformers import AlbertModel, AlbertForSequenceClassification, AlbertTokenizer, AdamW, get_linear_schedule_with_warmup

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertModel.from_pretrained('albert-base-v2')

if torch.cuda.is_available():    
    device = torch.device("cuda:0")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

model.to(device)


def albert_embedder(data_file, save_file_parent, save_file_reply, max_len, batch_size):
    sentences = []
    with open(data_file) as csv_file:
        reader = csv.reader(csv_file)
        next(reader) # skip first row
        for row in reader:
            sentences.append([row[2], row[1]])
    
    parent_sents = []
    parent_masks = []
    parent_embeddings = []
    reply_sents = []
    reply_masks = []
    reply_embeddings = []
    count = 0
    prev_count = 0
    for sent in tqdm(sentences):
        p_inputs = tokenizer.encode_plus(sent[0], max_length=max_len//2, pad_to_max_length=True, 
                                        return_attention_mask=True, return_tensors = 'pt')
        r_inputs = tokenizer.encode_plus(sent[1], max_length=max_len//2, pad_to_max_length=True, 
                                        return_attention_mask=True, return_tensors = 'pt')
        parent_sents.append(p_inputs["input_ids"])
        parent_masks.append(p_inputs["attention_mask"])
        reply_sents.append(r_inputs["input_ids"])
        reply_masks.append(r_inputs["attention_mask"])

        if len(parent_sents) == batch_size:
            count += batch_size
            p_input_ids = torch.cat(parent_sents, dim=0)
            p_attention_mask = torch.cat(parent_masks, dim=0) 
            p_outputs = model(input_ids=p_input_ids.to(device), attention_mask=p_attention_mask.to(device))[0]
            p_ouputs_val = p_outputs.detach().cpu()
            parent_embeddings += [out for out in p_ouputs_val]
            
            r_input_ids = torch.cat(reply_sents, dim=0)
            r_attention_mask = torch.cat(reply_sents, dim=0)
            r_outputs = model(input_ids=r_input_ids.to(device), attention_mask=r_attention_mask.to(device))[0]
            r_outputs_val = r_outputs.detach().cpu()
            reply_embeddings += [out for out in r_outputs_val]

            parent_sents = []
            parent_masks = []

            reply_sents = []
            reply_masks = []
        
        if (count - prev_count) > 1000:
            print("Writing to file")
            with h5py.File(save_file_parent, "w") as hf:
                for i in range(count - prev_count):
                    hf.create_dataset(str(i), data=parent_embeddings[i])
                hf.close()
            
            with h5py.File(save_file_reply, "w") as hf:
                for i in range(count - prev_count):
                    hf.create_dataset(str(i), data=reply_embeddings[i])
                hf.close()
            
            parent_embeddings = []
            reply_embeddings = []
            prev_count = count
        
    if parent_masks:
        p_input_ids = torch.cat(parent_sents, dim=0)
        p_attention_mask = torch.cat(parent_masks, dim=0)
        p_outputs = model(input_ids=p_input_ids.to(device), attention_mask=p_attention_mask.to(device))[0]
        p_ouputs_val = p_outputs.detach().cpu()
        parent_embeddings += [out for out in p_ouputs_val]

        r_input_ids = torch.cat(reply_sents, dim=0)
        r_attention_mask = torch.cat(reply_sents, dim=0)
        r_outputs = model(input_ids=r_input_ids.to(device), attention_mask=r_attention_mask.to(device))[0]
        r_outputs_val = r_outputs.detach().cpu()
        reply_embeddings += [out for out in r_outputs_val]
    
    print("Writing to file")
    with h5py.File(save_file_parent, "w") as hf:
        for i in range(count - prev_count):
            hf.create_dataset(str(i), data=parent_embeddings[i])
        hf.close()
    
    with h5py.File(save_file_reply, "w") as hf:
        for i in range(count - prev_count):
            hf.create_dataset(str(i), data=reply_embeddings[i])
        hf.close()
    
    return parent_embeddings, reply_embeddings
    
