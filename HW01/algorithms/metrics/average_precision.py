import numpy as np
from .precision import precision_at_k

def average_precision(relevance_query: list):
    relevance_query = np.array(relevance_query)
    nb_relevant = np.sum(relevance_query == 1)
    nb_relevant_found = 0
    sum_precisionss = 0
    k = 0

    while nb_relevant != nb_relevant_found and k < len(relevance_query):
        if relevance_query[k] == 1:
            nb_relevant_found += 1
            sum_precisionss += precision_at_k(relevance_query, k+1)
        k += 1

    return sum_precisionss / nb_relevant if nb_relevant > 0 else 0
