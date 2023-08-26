# BERT_EXTSUM_NER (extractive summarization and named entity recognition)
This repository presents and compares BERT based models for extractive summarization, named entity recognition or both.  
  
This repository also present the influence of the summary/document ratio on performance.  
  
The datasets are CNN-DailyMail, NYT50 and part of the French wikipedia.  

## Clone project
```bash
git clone https://github.com/Baragouine/BERT_EXTSUM_NER.git
```

## Enter into the directory
```bash
cd BERT_EXTSUM_NER
```

## Create environnement
```bash
conda create --name BERT_EXTSUM_NER python=3.9
```

## Activate environnement
```bash
conda activate BERT_EXTSUM_NER
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Install nltk data
To install nltk data:
  - Open a python console.
  - Type ``` import nltk; nltk.download()```.
  - Download all data.
  - Close the python console.

## Data acquisition
  - CNN/DailyMail: See [https://github.com/Baragouine/SummaRuNNer/blob/master/README.md](https://github.com/Baragouine/SummaRuNNer/blob/master/README.md)
  - NYT50: See [https://github.com/Baragouine/HeterSUMGraph/blob/master/README.md](https://github.com/Baragouine/HeterSUMGraph/blob/master/README.md)
  - Wikipedia: See [https://github.com/Baragouine/HSG_ExSUM_NER/blob/master/README.md](https://github.com/Baragouine/HSG_ExSUM_NER/blob/master/README.md)

## Training
Run one of the notebooks below to train and evaluate the associated model:  
  - `01-train_camembert_ext_summary_and_ner.ipynb`: CAMEMBERT based model for both summarization and named entity recognition (for the French Wikipedia articles).
  - `02-train_camembert_ext_summary.ipynb`: CAMEMBERT based model for summarization only (for the French Wikipedia articles).
  - `03-train_camembert_ner.ipynb`: CAMEMBERT based model for entity recognition only (for the French Wikipedia articles).
  - `04-train_camembert_base_ccnet_ext_summary_and_ner.ipynb`: CAMEMBERT_BASE_CCNET based model for both summarization and named entity recognition (for the French Wikipedia articles).
  - `05-train_camembert_base_wikipedia_4gb_ext_summary_and_ner.ipynb`: CAMEMBERT_BASE_WIKIPEDIA based model for both summarization and named entity recognition (for the French Wikipedia articles).
  - `06-train_bertbase_ext_summary_CNNDailyMail.ipynb`: BERTBASE model for summarization only on CNN/DailyMail.
  - `07-train_bertbase_ext_summary_NYT50.ipynb`: BERTBASE model for summarization only on CNN/DailyMail.

## Result

