def average_precision(relevance_query: list):
    nb_relevant = relevance_query.count(1)
    nb_relevant_found = 0
    sum_precisions = 0
    k=0
    while nb_relevant!=nb_relevant_found:
        if relevance_query[k]==1:
            nb_relevant_found+=1
            sum_precisions += precision_at_k(relevance_query, k+1)
        k+=1
        
    print (sum_precisions/nb_relevant)
    return (sum_precisions/nb_relevant)
