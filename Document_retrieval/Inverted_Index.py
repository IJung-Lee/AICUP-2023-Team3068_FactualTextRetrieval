import re
import sqlite3
import pandas as pd
import jieba
import jieba.posseg as pseg

from tqdm import tqdm
from pandas import json_normalize

import warnings
warnings.filterwarnings('ignore')


class Doc:
    docid = 0
    tf = 0
    ld = 0
    def __init__(self, docid, tf, ld):
        self.docid = docid
        self.tf = tf
        self.ld = ld
    def __repr__(self):
        return(str(self.docid) + '\t' + str(self.tf) + '\t' + str(self.ld))
    def __str__(self):
        return(str(self.docid) + '\t' + str(self.tf) + '\t' + str(self.ld))

def clean_list(seg_list):
    cleaned_dict = {}
    n = 0
    for i in seg_list:
        i = i.strip().lower()
        if i != '' and (i not in stop_words):
            if not (re.search('[a-z]', i) and len(i)<=3):
                n = n + 1
                if i in cleaned_dict:
                    cleaned_dict[i] = cleaned_dict[i] + 1
                else:
                    cleaned_dict[i] = 1
    return n, cleaned_dict


def write_postings_to_db(postings_lists, db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute('''DROP TABLE IF EXISTS postings''')
    c.execute('''CREATE TABLE postings
                 (term TEXT PRIMARY KEY, df INTEGER, docs TEXT)''')

    for key, value in postings_lists.items():
        doc_list = '\n'.join(map(str,value[1]))
        t = (key, value[0], doc_list)
        c.execute("INSERT INTO postings VALUES (?, ?, ?)", t)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # 創建DB
    '''Create Title DB'''
    wiki = pd.read_csv('../Wiki/wiki_clean_doc.csv', keep_default_na=False, na_values=[' '], encoding='utf-8')
    title_lists = {}
    for i in tqdm(range(len(wiki))):
        words = re.split(r"_\(", wiki['Article'][i])
        if re.search(r"[\u4e00-\u9fa5|a-zA-Z]", words[0]):
            title = "".join(re.compile(u'[\u4E00-\u9FA5|a-zA-Z|0-9]').findall(words[0]))
            title = title.lower()
        docid = int(i)
        
        if title in title_lists:
            title_lists[title][0] = title_lists[title][0] + 1 # df++
            title_lists[title][1].append(docid)
        else:
            title_lists[title] = [1, [docid]] # [df, [Doc]]

    write_postings_to_db(title_lists, './Result/ir_title.db')


    '''Create Document DB'''
    jieba.load_userdict('./userdict.txt')
    stop_pseg = set()
    postings_lists = {}
    f = open('./stopwords.txt', encoding = 'utf-8')
    words = f.read()
    stop_words = set(words.split('\n'))

    Doc_L = 0
    for i in tqdm(range(len(wiki))):
        title = wiki['Article'][i]
        text = wiki['Text'][i]
        docid = int(i)
        words = pseg.cut(title + '。' + text)
        # words = pseg.cut(text)
        seg_list = [w.word for w in words if (w.flag.startswith('n') or w.flag.startswith('v') or w)]

        ld, cleaned_dict = clean_list(seg_list)
        Doc_L = Doc_L + ld
        for key, value in cleaned_dict.items():
            d = Doc(docid, value, ld)
            if key in postings_lists:
                postings_lists[key][0] = postings_lists[key][0] + 1 # df++
                postings_lists[key][1].append(d)
            else:
                postings_lists[key] = [1, [d]] # [df, [Doc]]
    write_postings_to_db(postings_lists, './Result/ir.db')