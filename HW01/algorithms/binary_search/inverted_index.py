import pandas as pd
import glob
from KafNafParserPy import KafNafParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np
from utils_processor.processor import Processor
import json
import logging
import time
from nltk import word_tokenize
class InvertedIndex:
    """
    A class for creating and managing an inverted index from text documents.
    """

    def __init__(self):
        """
        Initialize the InvertedIndex class and set up logging.
        """
        self.processor_ = Processor()
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        Set up and configure the logger.

        Returns:
            logging.Logger: Configured logger object.
        """
        logger = logging.getLogger('InvertedIndex')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler('inverted_index.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    @staticmethod
    def save_text_processed_json(list_text_processed: list):
        """
        Save processed text to a JSON file.

        Args:
            list_text_processed (list): A list of processed text documents.

        Returns:
            None

        Example:
            processed_texts = ["text1", "text2", "text3"]
            InvertedIndex.save_text_processed_json(processed_texts)
        """
        with open('text_processed.json', 'w') as file:
            json.dump(list_text_processed, file)

    def read_files(self, directory: str = "data/docs-raw-texts/") -> pd.DataFrame:
        """
        Read NAF files from a directory and extract relevant information.

        Args:
            directory (str): The directory containing NAF files. Default is "data/docs-raw-texts/".

        Returns:
            pd.DataFrame: A DataFrame containing the extracted information (identifier, text, and title).

        Example:
            inverted_index = InvertedIndex()
            df = inverted_index.read_files("path/to/naf/files/")
        """
        self.logger.info(f"Reading files from directory: {directory}")
        start_time = time.time()

        files = glob.glob(directory + "wes2015.d*.naf")
        data = []

        for file in files:
            identifier = file.split(".")[-2][-3:]  
            naf_parser = KafNafParser(file)
            raw_text = naf_parser.get_raw()
            title = naf_parser.root.find('nafHeader/fileDesc').get('title')
            data.append({"identifier": identifier, "text": raw_text, "title": title})

        dataFrame = pd.DataFrame(data)

        end_time = time.time()
        self.logger.info(f"Finished reading files. Time taken: {end_time - start_time:.2f} seconds")

        return dataFrame

    def replace_title_on_text(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the title from the text and clean up newline characters.

        Args:
            dataFrame (pd.DataFrame): Input DataFrame containing 'text' and 'title' columns.

        Returns:
            pd.DataFrame: Updated DataFrame with cleaned text.

        Example:
            inverted_index = InvertedIndex()
            df = inverted_index.read_files()
            cleaned_df = inverted_index.replace_title_on_text(df)
        """
        self.logger.info("Replacing titles in text and cleaning newlines")
        start_time = time.time()

        for i in range(len(dataFrame["text"])):
            if dataFrame["title"][i] in dataFrame["text"][i]:
                dataFrame["text"][i] = dataFrame["text"][i].replace(dataFrame["title"][i] + ".", "")
            dataFrame["text"][i] = dataFrame["text"][i].replace("\n", "")

        end_time = time.time()
        self.logger.info(f"Finished replacing titles. Time taken: {end_time - start_time:.2f} seconds")

        return dataFrame

    def apply_process(self, dataFrame: pd.DataFrame) -> tuple:
        """
        Apply text preprocessing to the DataFrame.

        Args:
            dataFrame (pd.DataFrame): Input DataFrame containing 'text' and 'title' columns.

        Returns:
            tuple: A tuple containing:
                - list: Processed text documents
                - pd.DataFrame: Original DataFrame
        """
        try:



            self.logger.info("Starting text preprocessing")
            start_time = time.time()

            text = dataFrame['text'] + " " + dataFrame['title']
            text = np.array(text).tolist()
            
            total_documents = len(text)
            text_processed = []
            
            for i, doc in enumerate(text, 1):
                progress = (i / total_documents) * 100
                self.logger.info(f"Processing document {i}/{total_documents} - {progress:.2f}% complete")

                processed_doc = word_tokenize(doc)
                processed_doc = self.processor_.to_lowercase(processed_doc)
                processed_doc = self.processor_.to_lowercase(processed_doc)
                processed_doc = self.processor_.remove_punctuation(processed_doc)
                processed_doc = self.processor_.remove_non_ascii(processed_doc)
                processed_doc = self.processor_.remove_stopwords(processed_doc)
                processed_doc = self.processor_.lemmatize_verbs(processed_doc)
                processed_doc = ' '.join(processed_doc)

                text_processed.append(processed_doc)
                
                # Log every 5% progress
                if i % max(1, total_documents // 20) == 0:
                    self.logger.info(f"Overall progress: {progress:.2f}%")

            end_time = time.time()
            self.logger.info(f"Finished text preprocessing. Time taken: {end_time - start_time:.2f} seconds")

            return text_processed, dataFrame
        except Exception as e:
            print(f"Error on apply_process: {e}")

    def apply_vectorizer_and_process(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        """
        Apply text vectorization and processing to the DataFrame.

        Args:
            dataFrame (pd.DataFrame): Input DataFrame containing 'text' and 'title' columns.

        Returns:
            pd.DataFrame: A DataFrame with vectorized text and document identifiers.

        Example:
            inverted_index = InvertedIndex()
            df = inverted_index.read_files()
            vectorized_df = inverted_index.apply_vectorizer_and_process(df)
        """
        self.logger.info("Applying vectorizer and processing")
        start_time = time.time()

        text_process, dataframe_object_vectorizer = self.apply_process(dataFrame)
        
        count_vect = CountVectorizer(lowercase=False)

        X = count_vect.fit_transform(text_process)

        vocabulario = count_vect.get_feature_names_out()

        dataframe_to_return = pd.DataFrame(X.toarray(), columns=[vocabulario])

        dataframe_to_return['identifier_files'] = dataframe_object_vectorizer['identifier'].values
        dataframe_to_return.columns = [col[0] if isinstance(col, tuple) else col for col in dataframe_to_return.columns]

        end_time = time.time()
        self.logger.info(f"Finished vectorization and processing. Time taken: {end_time - start_time:.2f} seconds")

        return dataframe_to_return

    def inverted_index(self, dataFrame: pd.DataFrame) -> dict:
        """
        Create an inverted index from the processed DataFrame.

        Args:
            dataFrame (pd.DataFrame): Input DataFrame containing vectorized text and document identifiers.

        Returns:
            dict: An inverted index dictionary where keys are terms and values are lists of document identifiers.

        Example:
            inverted_index = InvertedIndex()
            df = inverted_index.read_files()
            vectorized_df = inverted_index.apply_vectorizer_and_process(df)
            inv_index = inverted_index.inverted_index(vectorized_df)
        """
        self.logger.info("Creating inverted index")
        start_time = time.time()

        inverted_index_dicc = {}
        for i in dataFrame.columns.tolist():
            if i != "identifier_files":
                list_variable = [dataFrame["identifier_files"].iloc[j] for j in range(len(dataFrame[i])) if dataFrame[i].iloc[j] > 0]
                inverted_index_dicc[i] = sorted(list_variable)

        end_time = time.time()
        self.logger.info(f"Finished creating inverted index. Time taken: {end_time - start_time:.2f} seconds")

        return inverted_index_dicc

    def save_inverted_index(self, inverted_index: dict, filename: str = "inverted_index.json"):
        """
        Save the inverted index to a JSON file.

        Args:
            inverted_index (dict): The inverted index dictionary to save.
            filename (str): The name of the JSON file to save the index to. Default is "inverted_index.json".

        Returns:
            None
        """
        self.logger.info(f"Saving inverted index to {filename}")
        try:
            with open(filename, 'w') as file:
                json.dump(inverted_index, file, indent=4)
            self.logger.info(f"Inverted index successfully saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving inverted index to {filename}: {e}")

    def inverted_index_complete_pipeline(self) -> dict:
        """
        Execute the complete pipeline to create an inverted index.

        Returns:
            dict: An inverted index dictionary where keys are terms and values are lists of document identifiers.

        Example:
            inverted_index = InvertedIndex()
            result = inverted_index.inverted_index_complete_pipeline()
        """
        self.logger.info("Starting complete inverted index pipeline")
        overall_start_time = time.time()

        self.logger.info("Step 1: Reading files")
        dataFrame = self.read_files()

        self.logger.info("Step 2: Replacing titles in text")
        dataFrame = self.replace_title_on_text(dataFrame)

        self.logger.info("Step 3: Applying vectorizer and processing")
        dataFrame = self.apply_vectorizer_and_process(dataFrame)

        self.logger.info("Step 4: Creating inverted index")
        inverted_index_to_return = self.inverted_index(dataFrame)

        self.logger.info("Step 5: Saving inverted index")
        self.save_inverted_index(inverted_index_to_return)

        overall_end_time = time.time()
        self.logger.info(f"Completed inverted index pipeline. Total time taken: {overall_end_time - overall_start_time:.2f} seconds")
        
        return inverted_index_to_return
<<<<<<< HEAD
=======

# Ejecutar la pipeline completa y guardar el índice invertido
index = InvertedIndex()
inverted_index = index.inverted_index_complete_pipeline()

>>>>>>> 0f29366843fbf693f5cfae6ace96d744ff8dfac3
