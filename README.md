# Team3068_FactualTextRetrieval
T-Brain Team3068 Factual Text Retrieval
README.md檔案交代安裝配置環境，重要模塊輸出/輸入，以讓第三方用戶可以除錯、重新訓練與重現結果。 

### 請[點此](https://drive.google.com/drive/folders/1FNjZ5L3uTMsezUJE-jt15rAkCIfv0_40?usp=share_link)下載所需的Model和資料


## Requirements 

作業系統：Windows
CPU：12th GEN Intel(R) Core(TM) i9-12900K
GPU：NVIDIA GeForce RTX 3090 24G
語言：Python 3.10.6
主要函式庫：
    Jieba
    sqlite3
    Pytorch
    transformers
    sentence_transformers


## Usage

### Document Retrieval

至雲端下載ir.db和ir_title.db並放入Document_retrieval或執行下列程式碼生成ir.db和ir_title.db。
```
python Inverted_Index.py
```

執行並更改doc_retrieval.ipynb內的路徑。


