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
    modelPath = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\pretrained_vic\\hfl_pretraineds_0511sentBase_docArt_epoch50_0519"
    semanticTrain1Path = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\SemanticRankerData_eva\\claim_train_BM25F_v3.json"
    semanticTrain2Path = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\SemanticRankerData_eva\\claim_train2_BM25F_v3.json"
    semanticTest1Path = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\SemanticRankerData_eva\\claim_test_BM25F_v3.json"
    semanticTest2Path = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\SemanticRankerData_eva\\claim_private_test_BM25F_v3.json"
    semanticResultPath = "C:\\Users\\Howard\\Desktop\\Reva\\競賽\\vic\\sbert\\SemanticResult\\semanticRes_0511SBDA50e0519_BM25Fv3All_0530.jsonl"
    
    tokenizer = AutoTokenizer.from_pretrained(modelPath)
    model = AutoModel.from_pretrained(modelPath)
    model = model.to(device)
    model_ = SentenceTransformer(modelPath, device='cuda')
    data_list = json.load(open(semanticTrain1Path, 'r', encoding='utf-8')) + json.load(open(semanticTrain2Path, 'r', encoding='utf-8')) + json.load(open(semanticTest1Path, 'r', encoding='utf-8')) + json.load(open(semanticTest2Path, 'r', encoding='utf-8'))
   
    similar_evs = []
    for row in tqdm(data_list):
        claim = row['claim']
        ev_sents = []
        article_sents = []
        id_ = []
        for ev in row['evidence'].values():
            if ev['Text'] == []:
                ev_sents = ['None']
                id_.append('None')
                article_sents += [ev['Article']]
            else:
                for text in ev['Text']:
                    ev_sents += [text[1]]
                    id_.append(text[0])
            article_sents += [ev['Article'] for i in ev['Text']]

        sent2sim = {}

        for ev_sent in ev_sents:
            if ev_sent in sent2sim:
                continue

            sent2sim[ev_sent] = [row['id'],article_sents[ev_sents.index(ev_sent)],id_[ev_sents.index(ev_sent)],cosSimilarity(claim, ev_sent, model_)]
        sent2sim = list(sent2sim.items())
        sent2sim.sort(key=lambda s: s[1][3], reverse=True)
        similar_evs.append([ [s[0], s[1][0:3]] for s in sent2sim[:5] ])
        with open(semanticResultPath, 'a+', encoding='utf-8') as f:
            tmp = json.dumps([ [s[0], s[1][0:3]] for s in sent2sim[:5] ], ensure_ascii=False)
            print(tmp, file=f)


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
