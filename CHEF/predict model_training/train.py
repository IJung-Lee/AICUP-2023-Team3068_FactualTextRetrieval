import sys
import torch.optim as optim
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertTokenizer
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import numpy as np
import os 
import copy
import time, json
import datetime
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split, Subset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_recall_fscore_support
from collections import Counter
import csv
import pickle
import argparse
torch.cuda.empty_cache()

# ------------------------init parameters----------------------------
parser = argparse.ArgumentParser(description='Bert Classification For CHEF')
parser.add_argument('--cuda', type=str, default="0",help='appoint GPU devices')
parser.add_argument('--num_labels', type=int, default=3, help='num labels of the dataset')
parser.add_argument('--max_length', type=int, default=512, help='max token length of the sentence for bert tokenizer')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--initial_lr', type=float, default=5e-6, help='initial learning rate')
parser.add_argument('--initial_eps', type=float, default=1e-8, help='initial adam_epsilon')
parser.add_argument('--epochs', type=int, default=16, help='training epochs for labeled data')
parser.add_argument('--total_epochs', type=int, default=10, help='total epochs of the RL learning')
parser.add_argument('--seed_val', type=int, default=42, help='initial random seed value')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
device = torch.device("cuda")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    non_zero_idx = (labels_flat != 0)
    
    return np.sum(pred_flat[non_zero_idx] == labels_flat[non_zero_idx]) / len(labels_flat[non_zero_idx])

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def pre_processing(sentence_train, sentence_train_label, bert_type):
    input_ids = []
    attention_masks = []
    labels = []

    # Load tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    # pre-processing sentenses to BERT pattern
    for i in range(len(sentence_train)):
        encoded_dict = tokenizer.encode_plus(
            sentence_train[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=args.max_length,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            truncation=True
        )
        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(sentence_train_label[i])
    
    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0).to(device)
    attention_masks = torch.cat(attention_masks, dim=0).to(device)
    labels = torch.tensor(labels, device='cuda')
    # Combine the training inputs into a TensorDataset.
    train_dataset = TensorDataset(input_ids, attention_masks, labels)
    return train_dataset, tokenizer


def stratified_sample(dataset, ratio):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    sampled_indices = []
    rest_indices = []
    for indices in data_dict.values():
        random.shuffle(indices)
        sampled_indices += indices[0:int(len(indices) * ratio)]
        rest_indices += indices[int(len(indices) * ratio):len(indices)]
    return [Subset(dataset, sampled_indices), Subset(dataset, rest_indices)]

def drop1(dataset, max1num):
    data_dict = {}
    for i in range(len(dataset)):
        if not data_dict.get(dataset[i][2].item()):
            data_dict[dataset[i][2].item()] = []
        data_dict[dataset[i][2].item()].append(i)
    indices = data_dict[1]
    random.shuffle(indices)
    sampled_indices = []
    sampled_indices += indices[0:max1num]
    sampled_indices += (data_dict[0] + data_dict[2])
    return Subset(dataset, sampled_indices)

def prepareToTrain(sentence, sentence_label, bert_type, model_save_dir):
    dataset, tokenizer = pre_processing(sentence, sentence_label, bert_type)
    # split train and validation dataset
    val_dataset = Subset(dataset, [i for i in range(11620, len(dataset))])
    train_dataset = Subset(dataset, [i for i in range(11620)])
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=args.batch_size
    )

    # Load models
    model = BertForSequenceClassification.from_pretrained(
        bert_type,      # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--3
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model = nn.DataParallel(model)
    model = model.to(device)
    train_and_save_model(model, train_dataloader, val_dataloader, bert_type, model_save_dir)


def main(argv = None):
    oursWikiTrainPath = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\OURS\\CHEF_wiki_train_0511SBDA50e0519_BM25Fv3ALL_0607.json"
    oursWikiClaimCossimPath = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\OURS\\CHEF_wiki_claim_cossim_0511SBDA50e0519_BM25Fv3ALL_0607.json"
    datalist = json.load(open(oursWikiTrainPath, 'r', encoding='utf-8')) 
    labels = [row['label'] for row in datalist]
    
    print('===========Cos Similar begin===========')
    sentence = json.load(open(oursWikiClaimCossimPath, 'r', encoding='utf-8'))
    setRandomSeed()
    getDataSet(sentence, labels)
    print('===========Cos Similar end===========\n')

def setRandomSeed():
    # Set the seed value all over the place to make this reproducible.
    random.seed(args.seed_val)
    np.random.seed(args.seed_val)
    torch.manual_seed(args.seed_val)
    torch.cuda.manual_seed_all(args.seed_val)

def getDataSet(sentence, sentence_label):
    prepareToTrain(sentence, sentence_label, 'C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\pretrained_vic\\hfl_pretraineds_0511sentBase_docArt_epoch50_0519', './hfl_0511SBDA50e0519_epoch16_BM25Fv3All_0607/')

def train_and_save_model(model, train_dataloader, val_dataloader, bert_type='C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\pretrained_vic\\hfl_pretraineds_0511sentBase_docArt_epoch50_0519', model_save_dir='model_save'):
    predictionResultPath = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\PredictionResult\\all_prediction_0511SBDA50e0519_epoch16_BM25Fv3All_r_0607.pickle"
    # define the loss function
    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=args.initial_lr,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=args.initial_eps  # args.adam_epsilon  - default is 1e-8.
    )

    total_steps = len(train_dataloader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = 0,
        num_training_steps = total_steps
    )

    training_stats = []
    total_t0 = time.time()

    param_x = []
    param_y = []
    best_microf1 = 0
    best_macrof1 = 0
    best_recall = 0
    best_precision = 0
    best_prediction = None
    best_ground_truth = None
    for epoch_i in range(0, args.epochs):
        
        # ========================================
        #               Training
        # ========================================
        
        # Perform one full pass over the training set.
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode.
        model.train()

        # save mixup features
        all_logits = np.array([])
        all_ground_truth = np.array([])

        # For each batch of training data...
        epoch_params = []
        for step, batch in enumerate(train_dataloader):
            batch_params = np.array([])
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask, 
                labels=b_labels
            )
            loss, logits = outputs[0], outputs[1]
            total_train_loss += loss.sum().item()
            # Perform a backward to calculate the gradients.
            loss.sum().backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            scheduler.step()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            labels_flat = label_ids.flatten()
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

        # ========================================
        #               Validation
        # ========================================
        t0 = time.time()
        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])

        print('\ntrain data score:')
        for batch in train_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            with torch.no_grad():
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss, logits = outputs[0], outputs[1]
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.3%}".format(f1))
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.3%}".format(pre))
        print("   Recall (macro): {:.3%}".format(recall))
        print("       F1 (macro): {:.3%}".format(f1))

        print("")
        print("Running Validation...")
        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        all_prediction = np.array([])
        all_ground_truth = np.array([])
        all_logits = np.array([])
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Unpack this training batch from our dataloader.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            #b_domains = batch[3].to('cpu').numpy()
            with torch.no_grad():
                outputs = model(
                    b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask,
                    labels=b_labels
                )
                loss, logits = outputs[0], outputs[1]
            # Accumulate the validation loss.
            total_eval_loss += loss.sum().item()
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
            all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)
            if len(all_logits) == 0:
                all_logits = logits
            else:
                all_logits = np.concatenate((all_logits, logits), axis=0)

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(val_dataloader)
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)
        print('Validation Elapsed: {:}.'.format(validation_time))
        c = Counter()
        for pred in all_prediction:
            c[int(pred)] += 1
        print(c)
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='micro')
        print("       F1 (micro): {:.2%}".format(f1))
        microf1 = f1
        pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction, average='macro')
        print("Precision (macro): {:.2%}".format(pre))
        print("   Recall (macro): {:.2%}".format(recall))
        print("       F1 (macro): {:.2%}".format(f1))
        
        if f1 > best_macrof1:
            best_microf1 = microf1
            best_macrof1 = f1
            best_recall = recall
            best_precision = pre
            print('Above is best')
            # display every label's f1 score
            pre, recall, f1, _ = precision_recall_fscore_support(all_ground_truth, all_prediction)
            
            with open(predictionResultPath, 'wb') as f:
                pickle.dump(all_prediction, f)
            
            print("Precision:", pre)
            print("   Recall:", recall)
            print("       F1:", f1)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    print("Saving model to %s" % model_save_dir)

    model_to_save = model.module if hasattr(model, 'module') else model
    # Take care of distributed/parallel training
    model_to_save.save_pretrained(model_save_dir)


if __name__ == '__main__':
    sys.exit(main())
    