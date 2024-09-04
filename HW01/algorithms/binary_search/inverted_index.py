import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import json
import logging
import time
from algorithms.binary_search.text_processor import TextProcessor

class InvertedIndex:
    def __init__(self):
        self.logger = self.setup_logger()
        self.text_processor = TextProcessor()

    def setup_logger(self):
        logger = logging.getLogger('InvertedIndex')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        fh = logging.FileHandler('inverted_index.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def apply_vectorizer_and_process(self, dataFrame: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Applying vectorizer and processing")
        start_time = time.time()

        count_vect = CountVectorizer(lowercase=False)
        X = count_vect.fit_transform(dataFrame['text'])
        vocabulario = count_vect.get_feature_names_out()

        dataframe_to_return = pd.DataFrame(X.toarray(), columns=[vocabulario])
        dataframe_to_return['identifier_files'] = dataFrame['identifier'].values
        dataframe_to_return.columns = [col[0] if isinstance(col, tuple) else col for col in dataframe_to_return.columns]

        end_time = time.time()
        self.logger.info(f"Finished vectorization and processing. Time taken: {end_time - start_time:.2f} seconds")

        return dataframe_to_return

    def inverted_index(self, dataFrame: pd.DataFrame, occurrences: bool = False) -> dict:
        self.logger.info("Creating inverted index")
        start_time = time.time()

        inverted_index_dicc = {}
        for i in dataFrame.columns.tolist():
            if i != "identifier_files":
                list_variable = [dataFrame["identifier_files"].iloc[j] for j in range(len(dataFrame[i])) if dataFrame[i].iloc[j] > 0]
                
                if occurrences:
                    total_occurrences = sum(dataFrame[i])
                    clave = f"({i},{total_occurrences})"
                else:
                    clave = i

                inverted_index_dicc[clave] = sorted(list_variable)

        end_time = time.time()
        self.logger.info(f"Finished creating inverted index. Time taken: {end_time - start_time:.2f} seconds")

        return inverted_index_dicc

    def save_inverted_index(self, inverted_index: dict, filename: str = "inverted_index_without_ocurrences.json"):
        self.logger.info(f"Saving inverted index to {filename}")
        try:
            with open(filename, 'w') as file:
                json.dump(inverted_index, file, indent=4)
            self.logger.info(f"Inverted index successfully saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving inverted index to {filename}: {e}")

    def inverted_index_complete_pipeline(self, directory: str = "data/docs-raw-texts/", occurrences: bool = False) -> dict:
        self.logger.info("Starting complete inverted index pipeline")
        overall_start_time = time.time()

        self.logger.info("Step 1: Processing texts")
        processed_df = self.text_processor.process_texts(directory)

        self.logger.info("Step 2: Applying vectorizer and processing")
        vectorized_df = self.apply_vectorizer_and_process(processed_df)

        self.logger.info("Step 3: Creating inverted index")
        inverted_index_to_return = self.inverted_index(vectorized_df, occurrences)

        self.logger.info("Step 4: Saving inverted index")
        self.save_inverted_index(inverted_index_to_return)

        overall_end_time = time.time()
        self.logger.info(f"Completed inverted index pipeline. Total time taken: {overall_end_time - overall_start_time:.2f} seconds")
        
        return inverted_index_to_return