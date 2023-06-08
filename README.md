# Team3068_FactualTextRetrieval

## 模型、資料下載

| 模型 | Google下載 |
| --- | --- |
| `hfl_pretraineds_0511sentBase_document_article_epoch50_0519` | [連結](https://drive.google.com/drive/folders/1CbU0po4OXgTDoKnka3-5cmW95RqKXYMd?usp=share_link) |
| `hfl_0511SBDA50e0519_epoch16_BM25Fv3All_0607` | [連結](https://drive.google.com/drive/folders/1rnGel3ZZJ19icdBfYIXcIa9Mza7bt_oB?usp=share_link) |

| 資料| Google下載 |
| --- | --- |
| `Wiki` | [連結]( https://drive.google.com/drive/folders/1_BIDpD_AL2G-rUi9Z_KJ7vciI5eOY5qB?usp=share_link) |


- 將Wiki Folder放入執行資料夾
- 將ir.db和ir_title.db放入`Document_retrieval`資料夾
- 將pretrained的hfl_0511SBDA50e0519_epoch16_BM25Fv3All_0607放入執行資料夾
- 將pretrained的hfl_pretraineds_0511sentBase_document_article_epoch50_0519放入執行資料夾
## Requirements 

作業系統：Windows  
GPU：NVIDIA GeForce RTX 3090 24G  
語言：Python 3.10.6  
主要函式庫：  
- Jieba  
- sqlite3  
- Pytorch
- transformers
- sentence_transformers  


## Usage

### Document Retrieval

至雲端下載ir.db和ir_title.db並放入`Document_retrieval`中或執行下列程式碼生成ir.db和ir_title.db。
```
python Inverted_Index.py
```

執行`doc_retrieval.ipynb`，產生BM25F計算結果，結果將生成在Document_retrieval中。  


### Sentence Retrieval
執行`Sentence_retrieval`資料夾中的`Semantic_Ranker.py`來計算和claim相似度最高的五句句子。
```
python Semantic_Ranker.py
```
再執行`Sentence_retrieval`資料夾中的`get_sents.py`生成對應格式。
```
python get_sents.py
```

### Claim Validation

