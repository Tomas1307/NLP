import numpy as np

def dcg_at_k(relevance_query, k):
    relevance_query = np.array(relevance_query)[:k]
    # Implementamos max(i, 2) en el cálculo de descuentos
    discounts = np.log2(np.maximum(np.arange(1, k + 1), 2))
    dcg = np.sum(relevance_query / discounts)
    return dcg
