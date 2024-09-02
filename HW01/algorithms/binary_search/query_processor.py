import json
import glob
import logging
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from KafNafParserPy import KafNafParser
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from typing import List, Dict
from algorithms.binary_search.inverted_index import InvertedIndex
from algorithms.binary_search.binary_search import BinarySearch
from utils_processor.processor import Processor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    A class for processing and managing queries.
    """

    def __init__(self, query_directory: str = "data/queries-raw-texts/"):
        """
        Initialize the QueryProcessor.

        Args:
            query_directory (str): Directory containing query files.
        """
        self.query_directory = query_directory
        self.queries_df = None
        self.processor = Processor()
        logger.info("QueryProcessor initialized")

    def read_queries(self) -> pd.DataFrame:
        """
        Read query files and create a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing query identifiers and raw text.
        """
        files = glob.glob(self.query_directory + "wes2015.q*.naf")
        data = []

        for file in files:
            identifier = file.split(".")[-2][-3:]
            naf_parser = KafNafParser(file)
            raw_text = naf_parser.get_raw()
            data.append({"identifier": identifier, "query": raw_text})
        
        self.queries_df = pd.DataFrame(data)
        logger.info(f"Read {len(data)} queries")
        return self.queries_df

    def process_query(self, query: str) -> List[str]:
        """
        Process a single query through the preprocessing pipeline.

        Args:
            query (str): Raw query text.

        Returns:
            List[str]: Processed query as a list of tokens.
        """
        tokens = word_tokenize(query)
        tokens = self.processor.to_lowercase(tokens)
        tokens = self.processor.remove_punctuation(tokens)
        tokens = self.processor.remove_non_ascii(tokens)
        tokens = self.processor.remove_stopwords(tokens)
        tokens = self.processor.lemmatize_verbs(tokens)
        return tokens

    def process_queries(self) -> pd.DataFrame:
        """
        Process all queries in the DataFrame.

        Returns:
            pd.DataFrame: DataFrame with processed queries.
        """
        if self.queries_df is None:
            self.read_queries()

        self.queries_df["processed_query"] = self.queries_df["query"].apply(self.process_query)
        self.queries_df["query_list"] = self.queries_df["processed_query"].apply(lambda x: ' '.join(x).split())
        logger.info("Processed all queries")
        return self.queries_df

class SearchEngine:
    """
    A class for performing binary search on processed queries.
    """

    def __init__(self, inverted_index_path: str = "inverted_index.json"):
        """
        Initialize the SearchEngine.

        Args:
            inverted_index_path (str): Path to the inverted index JSON file.
        """
        with open(inverted_index_path) as file:
            self.inverted_index = json.load(file)
        self.binary_search = BinarySearch()
        logger.info("SearchEngine initialized")

    def generate_results_file(self, queries_df: pd.DataFrame, output_file: str):
        """
        Generate a results file based on binary search of processed queries.

        Args:
            queries_df (pd.DataFrame): DataFrame containing processed queries.
            output_file (str): Path to the output file.
        """
        self.binary_search.generate_results_file(queries_df, self.inverted_index, output_file)
        logger.info(f"Results file generated: {output_file}")

def main():
    """
    Main function to orchestrate the query processing and search operations.
    """
    query_processor = QueryProcessor()
    queries_df = query_processor.process_queries()

    search_engine = SearchEngine()
    output_file = "BSII-AND-queries_result.txt"
    search_engine.generate_results_file(queries_df, output_file)

    logger.info(f"Results file content:")
    with open(output_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()