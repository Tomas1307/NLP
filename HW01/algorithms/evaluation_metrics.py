import pandas as pd
from metrics.average_precision import average_precision
from metrics.dcg import dcg_at_k
from metrics.mean_average_precision import MAP
from metrics.ndcg import ndcg_at_k
from metrics.precision import precision_at_k, precision
from metrics.recall import recall_at_k

class EvaluationMetrics:
    """
    A class for calculating various evaluation metrics for information retrieval systems.
    """
     
    def __init__(self) -> None:
        """
        Initialize the EvaluationMetrics class.
        """
        pass

    @staticmethod
    def precision(relevance_query: list) -> float:
        """
        Calculate the precision for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.

        Returns:
            float: The calculated precision.
        """
        return precision(relevance_query)
    
    @staticmethod
    def precision_at_k(relevance_query: list, k: int) -> float:
        """
        Calculate the precision at k for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.
            k (int): The number of top documents to consider.

        Returns:
            float: The calculated precision at k.
        """
        return precision_at_k(relevance_query, k)
    
    @staticmethod
    def recall_at_k(relevance_query: list, number_relevant_docs: int, k: int) -> float:
        """
        Calculate the recall at k for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.
            number_relevant_docs (int): The total number of relevant documents.
            k (int): The number of top documents to consider.

        Returns:
            float: The calculated recall at k.
        """
        return recall_at_k(relevance_query, number_relevant_docs, k)
    
    @staticmethod
    def average_precision(relevance_query: list) -> float:
        """
        Calculate the average precision for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.

        Returns:
            float: The calculated average precision.
        """
        return average_precision(relevance_query)
    
    @staticmethod
    def mean_average_precision(relevance_query: list) -> float:
        """
        Calculate the mean average precision for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.

        Returns:
            float: The calculated mean average precision.
        """
        return MAP(relevance_query)
    
    @staticmethod
    def dcg_at_k(relevance_query: list, k: int) -> float:
        """
        Calculate the Discounted Cumulative Gain (DCG) at k for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.
            k (int): The number of top documents to consider.

        Returns:
            float: The calculated DCG at k.
        """
        return dcg_at_k(relevance_query, k)
    
    @staticmethod
    def ndcg_at_k(relevance_query: list, k: int) -> float:
        """
        Calculate the Normalized Discounted Cumulative Gain (NDCG) at k for a given relevance query.

        Args:
            relevance_query (list): A list of relevance scores for retrieved documents.
            k (int): The number of top documents to consider.

        Returns:
            float: The calculated NDCG at k.
        """
        return ndcg_at_k(relevance_query, k)