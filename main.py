import argparse
from data.data import get_dataloader_LSTM, get_dataloader_ALBERT
from transformers import AlbertTokenizer, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from models.attentionalLSTM import PairAttnLSTM
from LSTM_trainer import trainer as LSTMTrainer
from ALBERT_trainer import trainer as AlbertTrainer

def main(args):
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    assert args.model in ["albert-classifier", "attn-lstm", "cond-attn-lstm"]
    if args.model == "albert-classifier":
        train_dataloader, validation_dataloader = get_dataloader_ALBERT(tokenizer, args.data_file, args.batch, args.max_len)
        classifier = AlbertForSequenceClassification.from_pretrained('albert-base-v2')
        classifier.config.classifier_dropout_prob = 0.1
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
            {'params': [p for n, p in classifier.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': args.wd},
            
            # Filter for parameters which *do* include those.
            {'params': [p for n, p in classifier.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = args.lr, eps = args.adam_eps)
        # optimizer = AdamW(classifier.parameters(), lr = LR, eps = EPS)
        total_steps = len(train_dataloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        # weight decay and clip_grad_norm 10K data LR = 1e-5 WD = 1e-4 BATCH=32 by setting accummulate = 2
        classifier, history = AlbertTrainer(classifier, optimizer, scheduler, args.epochs, args.early_stop, train_dataloader, validation_dataloader, accumulation_steps=args.accumulate)

    else:
        if args.model == "attn-lstm":
            train_dataloader, validation_dataloader = get_dataloader_LSTM(tokenizer, args.data_file, args.batch, args.max_len)
            classifier = PairAttnLSTM(embedding_dim=768, hidden_dim=args.d_hid, num_layers=args.n_layer, label_size=args.n_label)
            optimizer_grouped_parameters = [{'params': [p for n, p in classifier.parameters()], 'weight_decay_rate': args.wd}]

            optimizer = AdamW(classifier.parameters(), lr = args.lr, eps = args.adam_eps)
        else:
            pass

        total_steps = len(train_dataloader) * args.epochs

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps = 0, # Default value in run_glue.py
                                                    num_training_steps = total_steps)
        print("Start training ...")
        print("Max epochs", args.epochs)
        print("Early Stop", args.early_stop)
        print("Batch Size", args.batch)
        print("Accumulate", args.accummulate)
        print("Learning Rate", args.lr)
        print("Weight Decay", args.wd)
        print("Max Sequene Length", args.max_len)
        print("LSTM Hidden Size", args.d_hid)
        print("LSTM Layers", args.n_layer)
        print()
        classifier, history = LSTMTrainer(classifier, optimizer, scheduler, args.epochs, args.early_stop, train_dataloader, validation_dataloader, accumulation_steps=args.accumulate)    

def get_args():
    """
    python3 main.py --data_file combine_balanced.csv --model attn-lstm  """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", help="enter the path to csv file contain reddit comments")
    parser.add_argument("--model", choices=["albert-classifier", "attn-lstm", "cond-attn-lstm"])
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--d_hid", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=1)
    parser.add_argument("--n_label", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early_stop", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
