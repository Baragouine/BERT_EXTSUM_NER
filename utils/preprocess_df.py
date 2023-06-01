from nltk.tokenize import LineTokenizer, sent_tokenize
import torch
from .preprocess_text import preprocess_text

# preprocess df
def preprocess_df(df, bert_tokenizer, padding_trunc_doc, doc_column_name="docs", labels_sum_column_name="labels_sum", entities_column_name=None, is_sep_n=False):
  nltk_line_tokenizer = LineTokenizer()
  sent_tokenizer = None

  if is_sep_n:
    sent_tokenizer = lambda x: nltk_line_tokenizer.tokenize(x)
  else:
    sent_tokenizer = sent_tokenize

  result = []
  for idx in df.index:
    if entities_column_name is not None:
      input_ids, attention_mask, labels_entities = preprocess_text(df[doc_column_name][idx], sent_tokenizer=sent_tokenizer, bert_tokenizer=bert_tokenizer, padding_trunc_doc=padding_trunc_doc, entities=entities_column_name)
      nb_sent = torch.sum(input_ids == bert_tokenizer.sep_token_id).item()
      result.append({"idx" : idx, "input_ids" : input_ids, "attention_mask": attention_mask, "labels_sum" : df[labels_sum_column_name][idx][:nb_sent], "labels_ner" : labels_entities[:input_ids.shape[0]]})
    else:
      input_ids, attention_mask = preprocess_text(df[doc_column_name][idx], sent_tokenizer=sent_tokenizer, bert_tokenizer=bert_tokenizer, padding_trunc_doc=padding_trunc_doc)
      nb_sent = torch.sum(input_ids == bert_tokenizer.sep_token_id).item()
      result.append({"idx" : idx, "input_ids" : input_ids, "attention_mask": attention_mask, "labels" : df[labels_sum_column_name][idx][:nb_sent]})

  return result