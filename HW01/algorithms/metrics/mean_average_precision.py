def MAP(relevance_query: list):
    map_value = 0
    for i in range (len(relevance_query)):
        map_value+=average_precision(relevance_query[i])
    print (map_value/len(relevance_query))
    return (map_value/len(relevance_query))
