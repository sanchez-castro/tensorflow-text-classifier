
# Pre-processing data: create our tokenizer class
from tensorflow.keras.preprocessing import text

class TextPreprocessor(object):
  def __init__(self, vocab_size):
    self._vocab_size = vocab_size
    self._tokenizer = None
  
  def create_tokenizer(self, text_list):
    """
    This class allows to vectorize a text corpus, by turning each text into either a sequence of 
    integers (each integer being the index of a token in a dictionary) or into a vector where the 
    coefficient for each token could be binary, based on word count, based on tf-idf.
    """
    tokenizer = text.Tokenizer(num_words=self._vocab_size)
    tokenizer.fit_on_texts(text_list)
    self._tokenizer = tokenizer

  def transform_text(self, text_list):
    text_matrix = self._tokenizer.texts_to_matrix(text_list)
    return text_matrix
