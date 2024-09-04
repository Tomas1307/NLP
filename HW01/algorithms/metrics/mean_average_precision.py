import numpy as np
from .average_precision import average_precision

def MAP(relevance_queries: list):
    """
    Función que calcula la métrica ndcg para MEAN AVERAGE PRECISION
    Args
    relevance_queries: Lista de vectores binarios, cada uno representa un vector de resultado de la consulta
    """
    relevance_queries = [np.array(query) for query in relevance_queries]
    map_value = np.mean([average_precision(query) for query in relevance_queries])
    return map_value
