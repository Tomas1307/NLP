import json
import logging
import pandas as pd
from algorithms.binary_search.binary_search import BinarySearch
from algorithms.binary_search.query_processor import QueryProcessor
from algorithms.binary_search.inverted_index import InvertedIndex

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEngine:
    """
    A class for performing binary search on processed queries.
    """

    def __init__(self, inverted_index_path: str = "inverted_index_without_ocurrences.json"):
        """
        Initialize the SearchEngine.

        Args:
            inverted_index_path (str): Path to the inverted index JSON file.
        """
        try:
            with open(inverted_index_path) as file:
                self.inverted_index = json.load(file)
        except Exception as e:
            index = InvertedIndex()
            self.inverted_index = index.inverted_index_complete_pipeline()
            
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

