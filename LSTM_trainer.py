import torch
import numpy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import time
from livelossplot import PlotLosses
from tqdm import tqdm
import random
from transformers import AlbertModel, AlbertTokenizer, AdamW
from sklearn.metrics import precision_recall_fscore_support

if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def trainer(classifier, optimizer, scheduler, epochs, early_stop, train_dataloader, validation_dataloader, save_file, seed_val=0, accumulation_steps=1):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        classifier = nn.DataParallel(classifier)
    classifier.to(device)

    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    embedder = AlbertModel.from_pretrained('albert-base-v2')
    embedder.to(device)

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    best = (np.inf, -1, -np.inf, None, None)

    liveloss = PlotLosses()
    LossHistory = []
    val_step = 0
    for epoch_i in range(0, epochs):
        logs = {}
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print("Global Learning Rate", optimizer.param_groups[0]["lr"])
        print('Training...')
        classifier.train()
        epoch_loss = 0.
        start = time.time()
        classifier.zero_grad()
        for step, batch in enumerate(train_dataloader):
            b_inputs_c = batch[0].to(device)
            b_inputs_r = batch[1].to(device)
            b_mask_c = batch[2].to(device)
            b_mask_r = batch[3].to(device)
            b_labels = batch[4].to(device)
            
            x_c = embedder(input_ids=b_inputs_c, attention_mask=b_mask_c)[0]
            x_r = embedder(input_ids=b_inputs_r, attention_mask=b_mask_r)[0]
            loss, logits = classifier(x_c.permute(1,0,2), x_r.permute(1,0,2), b_labels)

            if torch.cuda.device_count() > 1:
                loss = loss.sum()

            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                classifier.zero_grad()
            
            batch_loss = loss.cpu().item()
            epoch_loss += loss.cpu().item()

            if (step % 100) == 0:
                print("Step %i with loss %.3f elapsed time %.3f" % (step, batch_loss, time.time() - start))
                # writer.add_scalar("Loss/train", epoch_loss/(step+1), global_step)
                # writer.flush()
            
        print("Step %i with loss %f elapsed time %f" % (step, batch_loss, time.time() - start))
        print('Evaluating...')
        classifier.eval()
        dev_loss = 0.
        total_eval_accuracy = 0.
        y_preds = None
        y_true = None
        for batch in validation_dataloader:
            b_inputs_c = batch[0].to(device)
            b_inputs_r = batch[1].to(device)
            b_mask_c = batch[2].to(device)
            b_mask_r = batch[3].to(device)
            b_labels = batch[4].to(device)
                
            with torch.no_grad():
                x_c = embedder(input_ids=b_inputs_c, attention_mask=b_mask_c)[0]
                x_r = embedder(input_ids=b_inputs_r, attention_mask=b_mask_r)[0]
                loss, logits = classifier(x_c.permute(1,0,2), x_r.permute(1,0,2), b_labels)
                if torch.cuda.device_count() > 1:
                    loss = loss.sum()
                    
            dev_loss += loss.cpu().item()
            label_ids = b_labels.cpu().numpy()
            logits = logits.detach().cpu().numpy()
            total_eval_accuracy += flat_accuracy(logits, label_ids)           
            if y_preds is None:
                y_preds = np.argmax(logits, axis=1)
                y_true = label_ids
            else:
                y_preds = np.concatenate((y_preds, np.argmax(logits, axis=1)))
                y_true = np.concatenate((y_true, label_ids))
        
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        f1_score_1 = precision_recall_fscore_support(y_true, y_preds, average="binary")
        f1_score_0 = precision_recall_fscore_support(y_true, y_preds, average="binary", pos_label=0)

        print("Epoch %i with dev loss %f and dev accuracy %f" % (epoch_i + 1, dev_loss, avg_val_accuracy))

        logs["val_loss"] = dev_loss / len(validation_dataloader)
        logs["loss"] = epoch_loss / len(train_dataloader)
        logs["val_accuracy"] = avg_val_accuracy
        liveloss.update(logs)
        LossHistory.append(logs["loss"])
        liveloss.send()

        if(val_step-best[1] >= early_stop and best[0] < dev_loss):
            print("early_stopping, epoch:", epoch_i+1)
            print("Final dev loss %f Final Train Loss %f Final dev accuracy %f" % (dev_loss, epoch_loss, avg_val_accuracy))
            print("Best dev loss %f Best dev accuracy %f" % (best[0], best[2]))
            print("F1_score Sarcasm ", f1_score_1)
            print("F1_score Non-Sarcasm ", f1_score_0)

            return classifier
        elif(best[0] > dev_loss):
            best = (dev_loss, val_step, avg_val_accuracy, f1_score_1, f1_score_0)
            torch.save(classifier.state_dict(), save_file)
            
        val_step += 1
        classifier.train()

        
    print("Final dev loss %f Final Train Loss %f Final dev accuracy %f" % (dev_loss, epoch_loss, avg_val_accuracy))
    print("Best dev loss %f Best dev accuracy %f" % (best[0], best[2]))
    print("F1_score Sarcasm ", f1_score_1)
    print("F1_score Non-Sarcasm ", f1_score_0)
    
    return classifier