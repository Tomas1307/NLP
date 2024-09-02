import json
import logging
import pandas as pd
from algorithms.binary_search.binary_search import BinarySearch
from algorithms.binary_search.query_processor import QueryProcessor


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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