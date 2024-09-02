import numpy

def precision(relevance_query: list):
        all_docs = len(relevance_query)
        pertinent = relevance_query.count(1)
        print(pertinent/all_docs)
        return (pertinent/all_docs)
    
def precision_at_k(relevance_query: list ,k: int):
    query_at_k = relevance_query[:k]
    all_docs= len(query_at_k)
    relevant = query_at_k.count(1)
    return (relevant/all_docs)
