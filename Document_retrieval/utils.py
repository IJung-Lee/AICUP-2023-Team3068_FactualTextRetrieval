import os
import re
import json
import numpy as np
import pandas as pd
from pandas import json_normalize
import warnings
warnings.filterwarnings('ignore')


def clean_space(text):
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    replace_list = match_regex.findall(text)
    order_list = sorted(replace_list,key=lambda i:len(i),reverse=True)
    for i in order_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text


def clean(text):
    if len(text) <= 1: pass
    elif re.search(r'(NOTOC)', text): pass  
    elif re.search(r'([0-9]+px)$', text): pass  
    elif re.search(r'[=]$', text): pass
    elif re.search(r'(jpg|png|bmp|svg|gif|tiff)$', text, re.IGNORECASE): pass
    elif re.search(r'^(infobox)', text, re.IGNORECASE): pass
    else: 
        return text
    

def wiki_num_sentence(folder_path):
    file = os.listdir(folder_path)
    files = []
    for filename in file:
        files.append(filename)   
    files = sorted(files)
    '''句子'''
    wikiALL = dict()
    for data in files:
        if data == '/.DS_Store':
            continue
        df = pd.read_json(path_or_buf=folder_path+'/'+data, lines=True)
        df.set_index('id' , inplace=True)
        df = df.fillna(0)
        wiki = df.to_dict('index')
        for i in wiki:
            for j in wiki[i]['lines'].split('\n'):
                if len(j)>5 and re.search(r'[\u4e00-\u9fa5_a-zA-Z]', j.split('\t', 2)[1]):
                    text = clean(j.split('\t', 2)[1])
                    try:
                        wikiALL[(i, int(j.split('\t', 2)[0]))] = clean_space(text)
                    except:pass
    wiki_df = pd.DataFrame(list(wikiALL.keys()), columns=['Article', 'Num'])
    wiki_df['Text'] = list(wikiALL.values())
    wiki_df.to_csv('../Wiki/wiki_clean.csv', index=False )
    return wiki_df

def wiki_doc(folder_path):
    file = os.listdir(folder_path)
    files = []
    for filename in file:
        files.append(filename)   
    files = sorted(files)
    '''文章'''
    wikiALL = dict()
    for data in files:
        df = pd.read_json(path_or_buf=folder_path+'/'+data, lines=True)
        df.set_index('id' , inplace=True)
        df = df.fillna(0)
        wiki = df.to_dict('index')

        for i in wiki:
            text = ""
            for j in wiki[i]['lines'].split('\n'):
                if len(j)>5 and re.search(r'[\u4e00-\u9fa5_a-zA-Z]', j.split('\t', 2)[1]):
                    try:
                        text+=(clean_space(clean(j.split('\t', 2)[1])))
                    except:pass
            wikiALL[i] = text
    wiki_doc = pd.DataFrame(list(wikiALL.keys()), columns=['Article'])
    wiki_doc['Text'] = list(wikiALL.values())
    wiki_doc.to_csv('../Wiki/wiki_clean_doc.csv', index=False )
    return wiki_doc


def wiki_arctext_doc(folder_path):
    file = os.listdir(folder_path)
    files = []
    for filename in file:
        files.append(filename)   
    files = sorted(files)
    '''斷句文章'''
    wikiALL = dict()
    for data in files:
        df = pd.read_json(path_or_buf=folder_path+'/'+data, lines=True)
        df.set_index('id' , inplace=True)
        df = df.fillna(0)
        wiki = df.to_dict('index')
        for i in wiki:
            text = []
            for j in wiki[i]['lines'].split('\n'):
                if len(j)>5 and re.search(r'[\u4e00-\u9fa5_a-zA-Z]', j.split('\t', 2)[1]):
                    try:
                        text.append(clean_space(clean(j.split('\t', 2)[1])))
                    except:pass
            wikiALL[i] = text
    wiki_doc = pd.DataFrame(list(wikiALL.keys()), columns=['Article'])
    wiki_doc['Text'] = list(wikiALL.values())
    wiki_doc.to_csv('../Wiki/wiki_clean_arctext_doc.csv', index=False )
    return wiki_doc


def wiki_numtext_doc(folder_path):
    file = os.listdir(folder_path)
    files = []
    for filename in file:
        files.append(filename)   
    files = sorted(files)
    '''帶num文章'''
    wikiALL = dict()
    for data in files:
        df = pd.read_json(path_or_buf=folder_path+'/'+data, lines=True)
        df.set_index('id' , inplace=True)
        df = df.fillna(0)
        wiki = df.to_dict('index')
        for i in wiki:
            text = []
            for j in wiki[i]['lines'].split('\n'):
                if len(j)>5 and re.search(r'[\u4e00-\u9fa5_a-zA-Z]', j.split('\t', 2)[1]):
                    try:
                        text.append([int(j.split('\t', 2)[0]), clean_space(clean(j.split('\t', 2)[1]))])
                    except:pass
            wikiALL[i] = text
    wiki_doc = pd.DataFrame(list(wikiALL.keys()), columns=['Article'])
    wiki_doc['Text'] = list(wikiALL.values())
    wiki_doc.to_csv('../Wiki/wiki_clean_numtext_doc.csv', index=False)
    return wiki_doc


def new_userdict(wiki_doc_path, save_path):
    # wiki_doc_path: './Wiki/wiki_clean_doc.csv'
    # save_path: './Doc_retrieval/userdict.txt'
    wiki = pd.read_csv(wiki_doc_path, keep_default_na=False, na_values=[' '], encoding='utf-8')
    Article = list(wiki['Article'])
    list_ner = []
    for word in Article:
        words = re.split(r"_\(", clean_space(word))
        if len(words[0])<=2 and re.search(r'[\u4e00-\u9fa5]', words[0]):
            if re.search(r'[0-9]+[年月]', words[0]):
                pass
            else:
                list_ner.append(words[0])
        if len(words[0])>=1 and len(words[0])<=15 and re.search(r"[\u4e00-\u9fa5|a-zA-Z]", word[0]):
            if re.search(r"[0-9]+[年月]", words[0]):
                pass
            else:
                text = "".join(re.compile(u'[\u4E00-\u9FA5|a-zA-Z|0-9]').findall(words[0]))
                list_ner.append(text)
    list_ner = list(set(list_ner))
    ner_file = open(save_path, "w",encoding = 'UTF-8')
    for ner in list_ner:
        ner_str = str(ner)
        ner_file.write(ner_str+" 100 nz \n")


def claim_evidence():
    claim1 = pd.read_json(path_or_buf='./Data/public_train_0316.jsonl', lines=True)
    claim2 = pd.read_json(path_or_buf='./Data/public_train_0522.jsonl', lines=True)
    wiki = pd.read_csv('../Wiki/wiki_clean.csv', keep_default_na=False, na_values=[' '])
    func = lambda x:[y for l in x for y in func(l)] if type(x[0]) == list else [x]

    text = []
    for i in claim1['evidence']:
        evs = func(i)
        txs = []
        for ev in evs:
            try:
                txs.append(wiki[(wiki['Article']==ev[2]) & (wiki['Num'] == ev[3])].Text.tolist()[0])
            except: pass
        text.append(txs)