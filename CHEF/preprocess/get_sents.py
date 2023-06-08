import os, sys
import json

def main():
    chef_test_train()
    claim_cossim()


def chef_test_train():
    data = json.load(open('..\document retrieval_data\claim_train_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_train2_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_test_BM25F_V3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_private_test_BM25F_v3.json', 'r', encoding='utf-8'))
    cossim_sents_lines = open('.\preprocessed_result\semanticRes_0511SBDA50e0519_BM25Fv3ALL_0530.jsonl', 'r', encoding='utf-8').readlines()[-len(data):]
    for index in range(len(data)):
        row = data[index]
        del data[index]['evidence']
        cossim_sents = json.loads(cossim_sents_lines[index].strip())
        cossim_sents = [t[0] for t in cossim_sents]
        row['cossim'] = cossim_sents

    # 放入要儲存的位置
    with open('.\preprocessed_result\CHEF_wiki_train_0511SBDA50e0519_BM25Fv3ALL_0607.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    

def claim_cossim():
    data = json.load(open('..\document retrieval_data\claim_train_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_train2_BM25F_v3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_test_BM25F_V3.json', 'r', encoding='utf-8')) + json.load(open('..\document retrieval_data\claim_private_test_BM25F_v3.json', 'r', encoding='utf-8'))
    cossim_sents_lines = open('.\preprocessed_result\semanticRes_0511SBDA50e0519_BM25Fv3ALL_0530.jsonl', 'r', encoding='utf-8').readlines()[-len(data):].readlines()[-len(data):]
    sent_list = []
    for index in range(len(data)):
        row = data[index]
        sents = json.loads(cossim_sents_lines[index].strip())
        sents = [t[0] for t in sents]
        sent = claim_evidences2bert_type(row['claim'], sents)
        sent_list.append(sent)
    # 放入要儲存的位置
    with open('.\preprocessed_result\CHEF_wiki_claim_cossim_0511SBDA50e0519_BM25Fv3ALL_0607.json', 'w', encoding='utf-8') as f:
        json.dump(sent_list, f, indent=2, ensure_ascii=False)


def claim_evidences2bert_type(claim: str, evidences: list):
    evlist = [claim] + evidences
    return f"[CLS] {' [SEP] '.join(evlist)} [SEP]"


if __name__ == '__main__':
    main()