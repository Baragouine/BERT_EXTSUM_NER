from nltk.tokenize import LineTokenizer, sent_tokenize

# Split one document
def split_doc(doc, sent_tokenizer, is_sep_n = False):
  result = doc
    
  # tokenize sentence
  result = sent_tokenizer(result)

  # lower
  result = [line.lower() for line in result]

  return result

# Split all document in the array
def split_all_docs(docs, is_sep_n = False):
  result = []

  nltk_line_tokenizer = LineTokenizer()
  sent_tokenizer = None

  if is_sep_n:
    sent_tokenizer = lambda x: nltk_line_tokenizer.tokenize(x)
  else:
    sent_tokenizer = sent_tokenize

  for doc in docs:
    result.append(split_doc(doc=doc, sent_tokenizer=sent_tokenizer, is_sep_n=is_sep_n))
  return result