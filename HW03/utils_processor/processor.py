from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer
import re, unicodedata
import logging
from text_to_num import text2num

class Processor:
    """
    Class that contains methods for text preprocessing, including tokenization, 
    normalization, punctuation removal, word-to-number conversion, and stemming.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Processor class with a logger and defines the total number of 
        steps in the preprocessing pipeline.
        """
        self.logger = logging.getLogger(__name__)
        self.total_steps = 6

    def remove_non_ascii(self, words):
        """
        Removes non-ASCII characters from a list of words.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with non-ASCII characters removed.
        """
        try:
            return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        except Exception as e:
            print(f"Error on remove_non_ascii: {e}")

    def to_lowercase(self, words):
        """
        Converts all words in the list to lowercase.
        
        Args:
            words (list): List of words to convert.
        
        Returns:
            list: List of words in lowercase.
        """
        try:
            return [word.lower() for word in words]
        except Exception as e:
            print(f"Error on to_lowercase: {e}")

    def remove_punctuation(self, words):
        """
        Removes punctuation from the list of words.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words without punctuation.
        """
        try:
            return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']
        except Exception as e:
            print(f"Error on remove_punctuation: {e}")
            
    def split_words(self, words):
        """
        Replaces underscores in words with spaces.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with underscores replaced by spaces.
        """
        try:
            return [word.replace("_", " ") for word in words]
        except Exception as e:
            print(f"Error in split_words: {e}")

    def words_to_numbers(self, words):
        """
        Converts words representing numbers to their numeric form.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with numbers converted to numeric form.
        """
        try:
            return [text2num(word, 'en') if word.isdigit() or word.isalpha() else word for word in words]
        except ValueError:
            return words

    def replace_digits(self, word):
        """
        Replaces digits with a placeholder string 'NUM'.
        
        Args:
            word (str): Word to process.
        
        Returns:
            str: 'NUM' if the word is a digit, otherwise returns the word unchanged.
        """
        return 'NUM' if word.isdigit() else word

    def remove_stopwords(self, words):
        """
        Removes stopwords from the list of words.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with stopwords removed.
        """
        try:
            stop_words = set(stopwords.words('english'))
            return [word for word in words if word not in stop_words]
        except Exception as e:
            print(f"Error on remove_stopwords: {e}")

    def stem_verbs(self, words):
        """
        Applies stemming to the list of words using the Snowball stemmer.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of stemmed words.
        """
        try:
            stemr = SnowballStemmer("english")
            return [stemr.stem(word) for word in words]
        except Exception as e:
            print(f"Error on stem_verbs: {e}")

    def preprocessing_pipeline(self, text):
        """
        Executes the full preprocessing pipeline, including tokenization, lowercase conversion, 
        punctuation removal, word-to-number conversion, digit replacement, stopword removal, 
        and stemming.
        
        Args:
            text (str): Input text to process.
        
        Returns:
            str: Processed text after the full preprocessing pipeline.
        """
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            # Tokenization
            self.logger.info("Step 1/6: Tokenizing text")
            text = word_tokenize(text)

            # Convert to lowercase
            self.logger.info("Step 2/6: Converting to lowercase")
            text = self.to_lowercase(text)

            # Remove punctuation
            self.logger.info("Step 3/6: Removing punctuation")
            text = self.remove_punctuation(text)

            # Convert words to numbers
            self.logger.info("Step 4/6: Converting words to numbers")
            text = self.words_to_numbers(text)

            # Replace digits
            self.logger.info("Step 5/6: Replacing digits")
            text = [self.replace_digits(word) for word in text]

            # Remove stopwords
            self.logger.info("Step 6/6: Removing stopwords")
            text = self.remove_stopwords(text)

            # Stemming
            self.logger.info("Step 7/6: Stemming verbs")
            text = self.stem_verbs(text)

            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(text)  # Return as a string
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            print(f"Error on preprocessing_pipeline: {e}")
            
    
    def preprocessing_pipeline_sentiments(self, text):
        """
        Executes a preprocessing pipeline specifically for sentiment analysis, including tokenization, 
        word splitting, lowercase conversion, punctuation removal, word-to-number conversion, 
        digit replacement, stopword removal, and stemming.
        
        Args:
            text (str): Input text to process.
        
        Returns:
            str: Processed text after the sentiment-specific preprocessing pipeline.
        """
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            # Tokenization
            self.logger.info("Step 1/7: Tokenizing text")
            text = word_tokenize(text)
            
            self.logger.info("Step 2/7: Splitting words")
            text = self.split_words(text)

            # Convert to lowercase
            self.logger.info("Step 3/7: Converting to lowercase")
            text = self.to_lowercase(text)

            # Remove punctuation
            self.logger.info("Step 4/7: Removing punctuation")
            text = self.remove_punctuation(text)

            # Convert words to numbers
            self.logger.info("Step 5/7: Converting words to numbers")
            text = self.words_to_numbers(text)

            # Replace digits
            self.logger.info("Step 6/7: Replacing digits")
            text = [self.replace_digits(word) for word in text]

            # Remove stopwords
            self.logger.info("Step 7/7: Removing stopwords")
            text = self.remove_stopwords(text)
            
            # Stemming
            self.logger.info("Step 7/7: Stemming verbs")
            text = self.stem_verbs(text)

            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(text)  # Return as a string
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            print(f"Error on preprocessing_pipeline: {e}")
