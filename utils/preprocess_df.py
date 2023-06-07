from nltk.tokenize import LineTokenizer, sent_tokenize
import torch
from .preprocess_text import preprocess_text

def split_labels_sum(labels_sum, list_input_ids, bert_tokenizer):
  res = []

  y = 0
  for input_ids in list_input_ids:
    n = torch.sum(torch.eq(input_ids, bert_tokenizer.sep_token_id)).item()
    res.append(labels_sum[y:y+n])

  return res

# preprocess df
def preprocess_df(df, bert_tokenizer, block_size, trunc_doc=-1, doc_column_name="docs", labels_sum_column_name="labels_sum", entities_column_name=None, is_sep_n=False):
  nltk_line_tokenizer = LineTokenizer()
  sent_tokenizer = None

  if is_sep_n:
    sent_tokenizer = lambda x: nltk_line_tokenizer.tokenize(x)
  else:
    sent_tokenizer = sent_tokenize

  result = []
  for idx in df.index:
    if entities_column_name is not None:
      list_input_ids, list_attention_mask, list_labels_entities = preprocess_text(df[doc_column_name][idx], sent_tokenizer=sent_tokenizer, bert_tokenizer=bert_tokenizer, trunc_doc=trunc_doc, block_size=block_size, entities=entities_column_name)
      result.append({"idx" : idx, "input_ids" : list_input_ids, "attention_mask": list_attention_mask, "labels_sum" : split_labels_sum(df[labels_sum_column_name][idx], list_input_ids, bert_tokenizer), "labels_ner" : list_labels_entities})
    else:
      list_input_ids, list_attention_mask = preprocess_text(df[doc_column_name][idx], sent_tokenizer=sent_tokenizer, bert_tokenizer=bert_tokenizer, trunc_doc=trunc_doc, block_size=block_size)
      result.append({"idx" : idx, "input_ids" : list_input_ids, "attention_mask": list_attention_mask, "labels" : split_labels_sum(df[labels_sum_column_name][idx], list_input_ids, bert_tokenizer)})

  return result