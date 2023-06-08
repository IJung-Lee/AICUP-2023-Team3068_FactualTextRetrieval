# Team3068_FactualTextRetrieval

### 請[點此](https://drive.google.com/drive/folders/1FNjZ5L3uTMsezUJE-jt15rAkCIfv0_40?usp=share_link)下載所需的Model和資料

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

### Claim Validation

