from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM
)
import torch
import os
import re
import sys
import copy
import random
import time, datetime
from time import sleep
import json, csv
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, 
    precision_recall_fscore_support,
    confusion_matrix
)
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses

from sklearn.metrics.pairwise import cosine_similarity



device = torch.device("cuda")
max_length = 256

def main():
    tokenizer = AutoTokenizer.from_pretrained("D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\sbert\\hfl_pretraineds_0511sentBase_document_article_epoch50_0519")
    model_ = SentenceTransformer("D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\sbert\\hfl_pretraineds_0511sentBase_document_article_epoch50_0519", device='cuda')
    data_list = json.load(open('D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\document retrieval_data\\claim_train_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\document retrieval_data\\claim_train2_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\document retrieval_data\\claim_test_BM25F_V3.json', 'r', encoding='utf-8')) + json.load(open('D:\\download\\比賽\\claim_private_test_BM25F_v3.json', 'r', encoding='utf-8'))
    data_list = data_list[9034:]
    similar_evs = []
    for row in tqdm(data_list):
        claim = row['claim']
        ev_sents = []
        article_sents = []
        id_ = []
        for ev in row['evidence'].values():
            if ev['Text'] == []:
                ev_sents += ['None']
                id_.append('None')
                article_sents += [ev['Article']]
                continue
            else:
                for text in ev['Text']:
                    ev_sents += [text[1]]
                    id_.append(text[0])
                    article_sents += [ev['Article']]

        # ev_sents += re.split(r'[？：。！（）.“”…\t\n]', row['content'])
        #for ev in row['evidence'].values():
            #tmp = list(lines['Text'][lines['Article'] == ev['Article']])
            #for i in range(len(tmp)):
                #ev_sents += [tmp[i]]
        #ev_sents = [sent for sent in ev_sents if len(sent) > 5]
        sent2sim = {}
        #print('ev_sents = ',ev_sents))

        for ev_sent in ev_sents:
            if ev_sent in sent2sim:
                continue
            #sent2sim[ev_sent] = cosSimilarity(claim, ev_sent, model, tokenizer)
            #sent2sim[ev_sent] = cosSimilarity(claim, ev_sent, model_)
        

            sent2sim[ev_sent] = [row['id'],article_sents[ev_sents.index(ev_sent)],id_[ev_sents.index(ev_sent)],cosSimilarity(claim, ev_sent, model_)]
        sent2sim = list(sent2sim.items())
        sent2sim.sort(key=lambda s: s[1][3], reverse=True)
        similar_evs.append([ [s[0], s[1][0:3]] for s in sent2sim[:5] ])
        with open('D:\\Vic\\GitHub\\Team3068_FactualTextRetrieval\\preprocessed_result\\semanticRes_0511SBDA50e0519_BM25Fv3ALL_0530.jsonl', 'a+', encoding='utf-8') as f:
            tmp = json.dumps([ [s[0], s[1][0:3]] for s in sent2sim[:5] ], ensure_ascii=False)
            print(tmp, file=f)


# def cosSimilarity(sent1, sent2, model, tokenizer):
#     encoded_dict = tokenizer.encode_plus(
#         sent1,  # Sentence to encode.
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=max_length,  # Pad & truncate all sentences.
#         padding='max_length',
#         return_attention_mask=True,  # Construct attn. masks.
#         return_tensors='pt',  # Return pytorch tensors.
#         truncation=True
#     )
#     input_ids = torch.tensor(encoded_dict['input_ids']).to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_ids)
#     last_hidden_state = outputs[0]
#     # CLS 对应的向量
#     sent1_vec = last_hidden_state[0][0].detach().cpu().numpy()
#     encoded_dict = tokenizer.encode_plus(
#         sent2,  # Sentence to encode.
#         add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
#         max_length=max_length,  # Pad & truncate all sentences.
#         padding='max_length',
#         return_attention_mask=True,  # Construct attn. masks.
#         return_tensors='pt',  # Return pytorch tensors.
#         truncation=True
#     )
#     input_ids = torch.tensor(encoded_dict['input_ids']).to(device)
#     model.eval()
#     with torch.no_grad():
#         outputs = model(input_ids)
#     last_hidden_state = outputs[0]
#     # CLS 对应的向量
#     sent2_vec = last_hidden_state[0][0].detach().cpu().numpy()
#     cos_sim = np.dot(sent1_vec, sent2_vec) / (np.linalg.norm(sent1_vec) * np.linalg.norm(sent2_vec))
#     return cos_sim.item()


def cosSimilarity(sent1, sent2, model):
    a = model.encode(sent1)
    b = model.encode(sent2)
    sim = cosine_similarity(a.reshape(1, -1),b.reshape(1, -1))
    return sim.item()


def tryMaxLen(tokenizer, model, sentences):
    max_len = 0
    for sent in sentences:
        input_ids = tokenizer.encode(sent, add_special_tokens=True)
        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))
    
    print(f"max sentence length = {max_len}")
    return max_len


if __name__ == '__main__':
    main()
