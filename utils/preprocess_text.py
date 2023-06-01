# URLs:
# * preprocessing 1 : https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0
# * preprocessing 2 : https://www.nltk.org/api/nltk.tokenize.html

import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
# Run the below line only the first time of running nltk
# nltk.download()

def preprocess_text(text, sent_tokenizer, bert_tokenizer, padding_trunc_doc, entities=None):
  # tokenize sentence
  text = sent_tokenizer(text)

  # Add [SEP]
  text = " [SEP] ".join(text)

  # tokenize with bert tokenizer
  inputs = bert_tokenizer.encode_plus(
    text,
    add_special_tokens=False,
    padding="max_length",
    max_length=padding_trunc_doc,
    return_tensors="pt",
    truncation=True
  )

  input_ids = inputs['input_ids'].squeeze()
  attention_mask = inputs['attention_mask'].squeeze()

  for i in range(input_ids.shape[0] - 1, -1, -1):
    if input_ids[i] != bert_tokenizer.pad_token:
      input_ids[i] = bert_tokenizer.sep_token_id
      break

  # Compute NER labels
  if entities is not None:
    linput_ids = input_ids.tolist()
    labels_entities = [0 for _ in range(input_ids.shape[0])]
    entities = sorted(entities, key=len, reverse=True)
    for entity in entities:
      entity_ids = bert_tokenizer.encode(entity, add_special_tokens=False)
      zeros = [0 for _ in entity_ids]
      for x, id in enumerate(linput_ids):
        if len(entity_ids) == len(linput_ids[x:x+len(entity_ids)]) and entity_ids == linput_ids[x:x+len(entity_ids)] and labels_entities[x:x+len(entity_ids)] == zeros:
          if len(entity_ids) >= 2:
            for i in range(len(entity_ids)):
              labels_entities[x+i] = 1#'C'
            labels_entities[x] = 1#'L'
            labels_entities[x+len(entity_ids) - 2] = 1#'R'
          else:
            labels_entities[x] = 1#'E'

    return input_ids, attention_mask, labels_entities
  
  return input_ids, attention_mask