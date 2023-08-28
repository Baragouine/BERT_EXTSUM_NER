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

### Impact of the summary/content ratio with CAMEMBERT_EXT on Wikipedia (limited-length ROUGE Recall)
| dataset | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:      |:-:      |:-:      |:-:      |  
| Wikipedia-0.5 |28.4 &plusmn; 0.0|8.1 &plusmn; 0.0|17.7 &plusmn; 0.0|  
| Wikipedia-high-25 |23.3 &plusmn; 0.0|6.4 &plusmn; 0.0|14.5 &plusmn; 0.0|  
| Wikipedia-low-25 |29.5 &plusmn; 0.0|10.3 &plusmn; 0.0|20.3 &plusmn; 0.0|  


&ast; Wikipedia-0.5: general geography, architecture town planning and geology French wikipedia articles with len(summary)/len(content) <= 0.5.  
&ast; Wikipedia-high-25: first 25% of general geography, architecture town planning and geology French wikipedia articles sorted by len(summary)/len(content) descending.  
&ast; Wikipedia-low-25: first 25% of general geography, architecture town planning and geology French wikipedia articles sorted by len(summary)/len(content) ascending.  

### Wikipedia-0.5 (limited-length ROUGE Recall)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L | ACCURACY |  
|:-:      |:-:      |:-:      |:-:      |:-:          |  
|camembert_ext_summary_and_ner|28.4 &plusmn; 0.0|8.1 &plusmn; 0.0|17.7 &plusmn; 0.0|0.997 &plusmn; 0.0|  
|camembert_ext_summary|28.4 &plusmn; 0.0|8.1 &plusmn; 0.0|17.7 &plusmn; 0.0|N/A|  
|camembert_ner|N/A|N/A|N/A|0.997 &plusmn; 0.0|  
|camembert_base_ccnet_ext_summary_and_ner|28.4 &plusmn; 0.0|8.1 &plusmn; 0.0|17.7 &plusmn; 0.0|0.997 &plusmn; 0.0|  
|camembert_base_wikipedia_4gb_ext_summary_and_ner|28.4 &plusmn; 0.0|8.1 &plusmn; 0.0|17.6 &plusmn; 0.0|0.997 &plusmn; 0.0|  

### CNN/DailyMail (full-length f1 rouge)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:    |:-:      |:-:      |:-:      |  
|SummaRuNNer(Nallapati)|39.6 &plusmn; 0.2|16.2 &plusmn; 0.2|35.3 &plusmn; 0.2|  
| BERT-base |31.4 &plusmn; 0.0|9.9 &plusmn; 0.0|19.2 &plusmn; 0.0|  

### NYT50 (limited-length ROUGE Recall)
| model | ROUGE-1 | ROUGE-2 | ROUGE-L |  
|:-:    |:-:      |:-:      |:-:      |  
| HeterSUMGraph (Wang) | 46.89 | 26.26 | 42.58 |
| BERT-base |38.4 &plusmn; 0.0|17.7 &plusmn; 0.0|26.8 &plusmn; 0.0|  
