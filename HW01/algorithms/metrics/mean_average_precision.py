import numpy as np
from .average_precision import average_precision

def MAP(relevance_queries: list):
    relevance_queries = [np.array(query) for query in relevance_queries]
    map_value = np.mean([average_precision(query) for query in relevance_queries])
    return map_value
