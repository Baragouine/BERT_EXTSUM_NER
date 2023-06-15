# URLs:
# * preprocessing 1 : https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0
# * preprocessing 2 : https://www.nltk.org/api/nltk.tokenize.html

import torch
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
# Run the below line only the first time of running nltk
# nltk.download()

def preprocess_text(text, sent_tokenizer, bert_tokenizer, block_size, entities=None, trunc_doc=-1):
  # list of input_ids and list of attention_mask
  list_input_ids = []
  list_attention_mask = []
  list_labels_entities = []

  # tokenize sentence
  tokenized_text = sent_tokenizer(text)

  if trunc_doc >= 0:
    tokenized_text = tokenized_text[:trunc_doc]

  # build list of input_ids without overflow for each input_ids
  y = 0
  while y < len(tokenized_text):
    text = (" " + bert_tokenizer.sep_token + " ").join(tokenized_text[y:]) + (" " + bert_tokenizer.sep_token + " ")

    inputs = bert_tokenizer.encode_plus(
      text,
      add_special_tokens=False,
      padding="max_length",
      max_length=block_size,
      return_tensors="pt",
      truncation=True
    )

    input_ids = inputs['input_ids'].squeeze()

    n = torch.sum(torch.eq(input_ids, bert_tokenizer.sep_token_id)).item()

    # Very long sentence, we will trunc it
    if n == 0:
      n = 1
      input_ids[-1] = bert_tokenizer.sep_token_id
      attention_mask = inputs['attention_mask'].squeeze()
    else:
      text = (" " + bert_tokenizer.sep_token + " ").join(tokenized_text[y:y+n]) + (" " + bert_tokenizer.sep_token + " ")

      inputs = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=False,
        padding="max_length",
        max_length=block_size,
        return_tensors="pt",
        truncation=True
      )

      input_ids = inputs['input_ids'].squeeze()
      attention_mask = inputs['attention_mask'].squeeze()

    list_input_ids.append(input_ids)
    list_attention_mask.append(attention_mask)
    y += n

  # Compute NER labels
  if entities is not None:
    for input_ids in list_input_ids:
      linput_ids = input_ids[torch.ne(input_ids, bert_tokenizer.pad_token_id) & torch.ne(input_ids, bert_tokenizer.sep_token_id)].tolist()
      labels_entities = [0 for _ in range(len(linput_ids))]
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
    
      list_labels_entities.append(labels_entities)

    return list_input_ids, list_attention_mask, list_labels_entities
  
  return list_input_ids, list_attention_mask