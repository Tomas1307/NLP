from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
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
        self.total_steps = 8  # Reflects the total number of steps in the pipeline.

    def remove_non_ascii(self, words):
        """
        Removes non-ASCII characters from a list of words by normalizing them.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with non-ASCII characters removed.
        """
        try:
            return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        except Exception as e:
            self.logger.error(f"Error in remove_non_ascii: {e}")
            return words

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
            self.logger.error(f"Error in to_lowercase: {e}")
            return words

    def remove_punctuation(self, words):
        """
        Removes punctuation from the list of words using regular expressions.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words without punctuation.
        """
        try:
            return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']
        except Exception as e:
            self.logger.error(f"Error in remove_punctuation: {e}")
            return words

    def split_words(self, words):
        """
        Replaces underscores in words with spaces, if necessary.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with underscores replaced by spaces.
        """
        try:
            return [word.replace("_", " ") for word in words]
        except Exception as e:
            self.logger.error(f"Error in split_words: {e}")
            return words

    def words_to_numbers(self, words):
        """
        Converts words representing numbers into their numeric form, where applicable.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with numbers converted to numeric form where possible.
        """
        numeric_words = []
        
        for word in words:
            # Verificar si la palabra es un número en palabras (ej. "one", "two", etc.)
            if re.match(r'^\d+$', word):  # Si es un dígito, se deja como está
                numeric_words.append(word)
            else:
                try:
                    # Solo intentar convertir si la palabra no es numérica pero puede ser un número en palabras
                    numeric_words.append(str(text2num(word, 'en')))
                except ValueError:
                    # Si no se puede convertir, mantener la palabra original
                    numeric_words.append(word)
        
        return numeric_words

    def replace_digits(self, word):
        """
        Replaces digit-only words with the placeholder string 'NUM'.
        
        Args:
            word (str): Word to process.
        
        Returns:
            str: 'NUM' if the word is a digit, otherwise the original word.
        """
        return 'NUM' if word.isdigit() else word

    def remove_stopwords(self, words):
        """
        Removes stopwords from the list of words using the NLTK stopwords corpus.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of words with stopwords removed.
        """
        try:
            stop_words = set(stopwords.words('english'))
            return [word for word in words if word not in stop_words]
        except Exception as e:
            self.logger.error(f"Error in remove_stopwords: {e}")
            return words

    def stem_verbs(self, words):
        """
        Applies stemming to the list of words using the SnowballStemmer.
        
        Args:
            words (list): List of words to process.
        
        Returns:
            list: List of stemmed words.
        """
        try:
            stemmer = SnowballStemmer("english")
            return [stemmer.stem(word) for word in words]
        except Exception as e:
            self.logger.error(f"Error in stem_verbs: {e}")
            return words
        

    def clean_gutenberg_text(self, text):
        """
        Cleans the Project Gutenberg text by removing all content before the line starting with
        "*** START OF THE PROJECT GUTENBERG EBOOK" or any variation.
        
        Args:
            text (str): The input text from a Project Gutenberg ebook.
            
        Returns:
            str: The cleaned text with content before the "START OF THE PROJECT GUTENBERG EBOOK" removed.
        """
        try:
            # Convert the entire text to lowercase for consistency
            if isinstance(text, list):  # Check if the input is a list
                text = ' '.join(text)    # Convert the list of tokens back into a string
                
            # Adjust the pattern to account for variations like spaces and mixed cases
            pattern = r'\*{1,3}\s*start of the project gutenberg ebook\s*.*\*{1,3}'
            
            #print("########################## STARTING CLEAN GUTENBERG TEXT ##########################")
            #print("text: ", text[:1500])

            # Step 1: Find the occurrence of the flexible pattern for "START OF THE PROJECT GUTENBERG EBOOK"
            start_match = re.search(pattern, text, re.IGNORECASE)
            #print("start_match: ", start_match)

            # Step 2: If the match is found, remove everything before it
            if start_match:
                start_index = start_match.start()
                cleaned_text = text[start_index:]
                cleaned_text = cleaned_text.replace("START OF THE PROJECT GUTENBERG EBOOK ","")
                #print("cleaned_text: ", cleaned_text[:1500])
                return cleaned_text
            else:
                raise ValueError("Start of the Project Gutenberg ebook not found")

        except Exception as e:
            #self.logger.error(f"Error cleaning Gutenberg text: {e}")
            return text

        
    def preprocessing_pipeline(self, text, index=None, total=None):
        """
        Executes the full preprocessing pipeline on the input text, which includes:
        - Gutenberg licensing removal
        - Tokenization
        - Lowercase conversion
        - Punctuation removal
        - Word-to-number conversion
        - Digit replacement
        - Stopword removal
        - Stemming
        
        Args:
            text (str): Input text to process.
            index (int): Current index of the text being processed (for logging).
            total (int): Total number of texts to process (for logging).
        
        Returns:
            str: Processed text after completing the preprocessing pipeline.
        """
        try:
            if index is not None and total is not None:
                self.logger.info(f"Processing text {index + 1}/{total}")

            # Step 1: Remove Project Gutenberg Licensing
            self.logger.info("Step 1/8: Cleaning Gutenberg text")
            text = self.clean_gutenberg_text(text)

            # Step 2: Tokenization
            self.logger.info("Step 2/8: Tokenizing text")
            words = word_tokenize(text)

            # Step 3: Convert to lowercase
            self.logger.info("Step 3/8: Converting to lowercase")
            words = self.to_lowercase(words)
            
            # Step 4: Remove punctuation
            self.logger.info("Step 4/8: Removing punctuation")
            words = self.remove_punctuation(words)

            # Step 5: Convert words to numbers
            self.logger.info("Step 5/8: Converting words to numbers")
            words = self.words_to_numbers(words)

            # Step 6: Replace digits
            self.logger.info("Step 6/8: Replacing digits")
            words = [self.replace_digits(word) for word in words]

            # Step 7: Remove stopwords
            self.logger.info("Step 7/8: Removing stopwords")
            words = self.remove_stopwords(words)

            # Optional: Stemming
            self.logger.info("Step 8/8: Stemming verbs (optional)")
            words = self.stem_verbs(words)

            self.logger.info("Preprocessing completed")
            return words  # Devolver como lista de tokens para Word2Vec
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline: {e}")
            return text.split()  # Si hay un error, devuelve la lista tokenizada básica
    
    def split_text_with_overlap(self, text, chunk_size=150, overlap_size=25):
        """
        Splits the text into chunks of a specified size, with overlap between chunks, based on words.
        
        Args:
            text (str): The input text to split.
            chunk_size (int): The number of words in each chunk.
            overlap_size (int): The number of overlapping words between chunks.
            
        Returns:
            list: A list of text chunks.
        """
        words = text.split()  # Split the text into words
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap_size):
            chunk = words[i:i + chunk_size]  # Select the chunk of words
            chunks.append(' '.join(chunk))  # Join the words to form the chunk as a string
        
        return chunks


        
            
    def preprocessing_pipeline_as_chunks(self, text, index=None, total=None, chunk_size=150, overlap_size=25):
        """
        Executes the full preprocessing pipeline on the input text, dividing the text into chunks
        of a specified length with overlap before processing each chunk.

        The pipeline includes:
        - Gutenberg licensing removal
        - Tokenization
        - Lowercase conversion
        - Punctuation removal
        - Word-to-number conversion
        - Digit replacement
        - Stopword removal
        - Stemming

        Args:
            text (str): Input text to process.
            index (int): Current index of the text being processed (for logging).
            total (int): Total number of texts to process (for logging).
            chunk_size (int): The length of each chunk (default 150).
            overlap_size (int): The overlap size between chunks (default 25).

        Returns:
            list: A list of processed text chunks after completing the preprocessing pipeline.
        """
        try:
            if index is not None and total is not None:
                self.logger.info(f"Processing text {index + 1}/{total}")

            # Step 1: Remove Project Gutenberg Licensing
            self.logger.info("Step 1/8: Cleaning Gutenberg text")
            text = self.clean_gutenberg_text(text)
            
            # Step 2: Divide text into chunks with overlap
            self.logger.info("Step 2/8: Splitting text into chunks")
            chunks = self.split_text_with_overlap(text, chunk_size, overlap_size)
            
            processed_chunks = []
            
            for chunk in chunks:
                # Step 3: Tokenization
                self.logger.info("Step 3/8: Tokenizing chunk")
                words = word_tokenize(chunk)

                # Step 4: Convert to lowercase
                self.logger.info("Step 4/8: Converting to lowercase")
                words = self.to_lowercase(words)
                
                # Step 5: Remove punctuation
                self.logger.info("Step 5/8: Removing punctuation")
                words = self.remove_punctuation(words)

                # Step 6: Convert words to numbers
                self.logger.info("Step 6/8: Converting words to numbers")
                words = self.words_to_numbers(words)

                # Step 7: Replace digits
                self.logger.info("Step 7/8: Replacing digits")
                words = [self.replace_digits(word) for word in words]

                # Step 8: Remove stopwords
                self.logger.info("Step 8/8: Removing stopwords")
                words = self.remove_stopwords(words)

                # Optional: Stemming
                self.logger.info("Step 9: Stemming verbs")
                words = self.stem_verbs(words)

                # Join the processed chunk back into a string and add to the list
                processed_chunks.append(' '.join(words))

            self.logger.info("Preprocessing completed for all chunks")
            return processed_chunks  # Devuelve una lista de fragmentos procesados
        except Exception as e:
            self.logger.error(f"Error in preprocessing_pipeline_as_chunks: {e}")
            return []
