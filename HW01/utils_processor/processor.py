from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re, unicodedata, contractions
from langdetect import detect_langs, LangDetectException
import logging
class Processor:
    def __init__(self) -> None:
        """
        Initializes the Processor class.
        """
        self.logger = logging.getLogger(__name__)
        self.total_steps = 7

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

    def remove_non_english_words(self, words):
        """
        Removes words that are not detected as English from a list of tokenized words.

        Args:
            words (list): A list of tokenized words.

        Returns:
            list: A list of words that are detected as English.
        """
        try:
            english_words = []
            for word in words:
                try:
                    # Detect the possible languages of the word
                    detected_langs = detect_langs(word)
                    # If English is one of the detected languages and has a high probability, keep the word
                    if any(lang.lang == 'en' and lang.prob > 0.9 for lang in detected_langs):
                        english_words.append(word)
                except LangDetectException:
                    # If language detection fails, assume it's not English and don't include it
                    continue
            return english_words
        except Exception as e:
            print(f"Error on remove_non_english_words: {e}")
            return words  # Return the original list if something goes wrong

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
            
            self.logger.info("Step 1/7: Tokenizing text")
            words = word_tokenize(text)
            
            self.logger.info("Step 2/7: Converting to lowercase")
            words = self.to_lowercase(words)
            
            self.logger.info("Step 3/7: Removing punctuation")
            words = self.remove_punctuation(words)
            
            self.logger.info("Step 4/7: Removing non-ASCII characters")
            words = self.remove_non_ascii(words)
            
            self.logger.info("Step 5/7: Removing non-English words")
            words = self.remove_non_english_words(words)
            
            self.logger.info("Step 6/7: Removing stopwords")
            words = self.remove_stopwords(words)
            
            self.logger.info("Step 7/7: Lemmatizing verbs")
            words = self.lemmatize_verbs(words)
            
            self.logger.info("Preprocessing pipeline completed")
            return ' '.join(words)
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            return text