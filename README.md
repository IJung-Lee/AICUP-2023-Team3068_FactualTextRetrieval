# Team3068_FactualTextRetrieval
T-Brain Team3068 Factual Text Retrieval
README.md檔案交代安裝配置環境，重要模塊輸出/輸入，以讓第三方用戶可以除錯、重新訓練與重現結果。 

### 請[點此](https://drive.google.com/drive/folders/1FNjZ5L3uTMsezUJE-jt15rAkCIfv0_40?usp=share_link)下載所需的Model和資料

- 將Wiki Folder放入執行資料夾。
- 將ir.db和ir_title.db放入`Document_retrieval`資料夾
- 將pretrained的`hfl_0511SBDA50e0519_epoch16_BM25Fv3All_0607`和`hfl_pretraineds_0511sentBase_document_article_epoch50_0519`放入...


## Requirements 

作業系統：Windows  
CPU：12th GEN Intel(R) Core(TM) i9-12900K  
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

執行doc_retrieval.ipynb，產生BM25F計算結果，結果將生成在Document_retrieval中。  


### Sentence Retrieval

### Claim Validation

