def dcg_at_k(relevance_query,k):
    dcg = 0
    for i in range(k):
        dcg = dcg + (relevance_query[i] / math.log2( max(i+1, 2)))
    print dcg
    return dcg
