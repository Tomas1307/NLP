from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, unicodedata
import logging

class Processor:

    def __init__(self) -> None:
        """
        Initializes the Processor class.
        """
        self.logger = logging.getLogger(__name__)
        self.total_steps = 6

    def remove_non_ascii(self, words):
        """
        Removes non-ASCII characters from a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of words with non-ASCII characters removed.
        """
        try:
            return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        except Exception as e:
            print(f"Error on remove_non_ascii: {e}")

    def to_lowercase(self, words):
        """
        Converts all characters to lowercase from a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of words converted to lowercase.
        """
        try:
            return [word.lower() for word in words]
        except Exception as e:
            print(f"Error on to_lowercase: {e}")

    def remove_punctuation(self, words):
        """
        Removes punctuation from a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of words with punctuation removed.
        """
        try:
            return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']
        except Exception as e:
            print(f"Error on remove_punctuation: {e}")

    def remove_stopwords(self, words):
        """
        Removes stop words from a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of words with stop words removed.
        """
        try:
            stop_words = set(stopwords.words('english'))
            return [word for word in words if word not in stop_words]
        except Exception as e:
            print(f"Error on remove_stopwords: {e}")

    def lemmatize_verbs(self, words):
        """
        Lemmatizes verbs in a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of lemmatized verbs.
        """
        try:
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(word, pos='v') for word in words]
        except Exception as e:
            print(f"Error on lemmatize_verbs: {e}")


    def preprocessing_pipeline(self, text):
        """
        Runs the full preprocessing pipeline on the input text.

        Args:
            text (str): The input text string to be processed.

        Returns:
            str: The processed text after tokenization, lowercasing, number replacement, punctuation removal, 
            non-ASCII character removal, non-English word removal, stopword removal, and lemmatization.
        """
        try:
            self.logger.info("Starting preprocessing pipeline")
            
            self.logger.info("Step 1/6: Tokenizing text")
            text = word_tokenize(text)

            
            self.logger.info("Step 2/6: Converting to lowercase")
            text = self.to_lowercase(text)

            
            self.logger.info("Step 3/6: Removing punctuation")
            text = self.remove_punctuation(text)

            
            self.logger.info("Step 4/6: Removing non-ASCII characters")
            text = self.remove_non_ascii(text)

            
            self.logger.info("Step 5/6: Removing stopwords")
            text = self.remove_stopwords(text)


            
            self.logger.info("Step 6/6: Lemmatizing verbs")
            text = self.lemmatize_verbs(text)

            
            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(text)
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            print(f"Error on preprocessing_pipeline: {e}")