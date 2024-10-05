from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import re, unicodedata
import logging
from text_to_num import text2num

class Processor:

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.total_steps = 6

    def remove_non_ascii(self, words):
        try:
            return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        except Exception as e:
            print(f"Error on remove_non_ascii: {e}")

    def to_lowercase(self, words):
        try:
            return [word.lower() for word in words]
        except Exception as e:
            print(f"Error on to_lowercase: {e}")

    def remove_punctuation(self, words):
        try:
            return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']
        except Exception as e:
            print(f"Error on remove_punctuation: {e}")
            
    def split_words(self, words):
        try:
            # Reemplazar "_" con " " en todas las palabras
            return [word.replace("_", " ") for word in words]
        except Exception as e:
            print(f"Error in split_words: {e}")
                

    def words_to_numbers(self, words):
        try:
            return [text2num(word, 'en') if word.isdigit() or word.isalpha() else word for word in words]
        except ValueError:
            return words

    def replace_digits(self, word):
        return 'NUM' if word.isdigit() else word

    def remove_stopwords(self, words):
        try:
            stop_words = set(stopwords.words('english'))
            return [word for word in words if word not in stop_words]
        except Exception as e:
            print(f"Error on remove_stopwords: {e}")

    def stem_verbs(self, words):
        try:
            stemr = SnowballStemmer("english")
            return [stemr.stem(word) for word in words]
        except Exception as e:
            print(f"Error on stem_verbs: {e}")

    def preprocessing_pipeline(self, text):
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            # Tokenización
            self.logger.info("Step 1/6: Tokenizing text")
            text = word_tokenize(text)

            # Convertir a minúsculas
            self.logger.info("Step 2/6: Converting to lowercase")
            text = self.to_lowercase(text)

            # Remover puntuación
            self.logger.info("Step 3/6: Removing punctuation")
            text = self.remove_punctuation(text)

            # Convertir palabras a números
            self.logger.info("Step 4/6: Converting words to numbers")
            text = self.words_to_numbers(text)

            # Reemplazar dígitos
            self.logger.info("Step 5/6: Replacing digits")
            text = [self.replace_digits(word) for word in text]

            # Remover stopwords
            self.logger.info("Step 6/6: Removing stopwords")
            text = self.remove_stopwords(text)

            # Stemming
            self.logger.info("Step 7/6: Stemming verbs")
            text = self.stem_verbs(text)

            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(text)  # Devolver como string
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            print(f"Error on preprocessing_pipeline: {e}")
            
    
    def preprocessing_pipeline_sentiments(self, text):
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            # Tokenización
            self.logger.info("Step 1/7: Tokenizing text")
            text = word_tokenize(text)
            
            self.logger.info("Step 2/7: Tokenizing text")
            text = self.split_words(text)

            # Convertir a minúsculas
            self.logger.info("Step 3/7: Converting to lowercase")
            text = self.to_lowercase(text)

            # Remover puntuación
            self.logger.info("Step 4/7: Removing punctuation")
            text = self.remove_punctuation(text)

            # Convertir palabras a números
            self.logger.info("Step 5/7: Converting words to numbers")
            text = self.words_to_numbers(text)

            # Reemplazar dígitos
            self.logger.info("Step 6/7: Replacing digits")
            text = [self.replace_digits(word) for word in text]

            # Remover stopwords
            self.logger.info("Step 7/7: Removing stopwords")
            text = self.remove_stopwords(text)

            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(text)  # Devolver como string
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            print(f"Error on preprocessing_pipeline: {e}")