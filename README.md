# User and Item Based Recommender System

This is tutorial from the book "Data Science From Scratch" by Joel Grus. In this tutorial author explains differences and approaches of two most common ways to build recommender systems: user-based collaborative filtering and item-based collaborative filtering.
- **User-based** approach finds users with similar items/iterests by utilizing similarity metric, such as cosine similarity, or distance metric (ex: KNN) and provides user of interest with new items suggested from similar users. 
- **Item-based** approach is trying to find similar items without involvement of the user.

## Importing Libraries


```python
from collections import defaultdict
```

## Data


```python
users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]
```

## 1. User-Based Collaborative Filtering

In the tutorial author decides to use cosine similarity metric.


```python
def dot(v, w):
    return sum([i*j for i,j in zip(v,w)])
```


```python
def cosine_similarity(v, w):
    return dot(v, w) / (dot(v, v) * dot(w, w))**(1/2)
```

First, we need to extract all unique items (interests) from all the users


```python
unique_interests = sorted(list({ interest
    for user_interests in users_interests
    for interest in user_interests }))
```


```python
print(f'Total users: {len(users_interests)}')
print(f'Total unique interests: {len(unique_interests)}')
```

    Total users: 15
    Total unique interests: 36


Then we build a matrix of items, where each user is represented as a vector, where `1` symbolizes presence of interest, `0` is absence. Final matrix will have shape of n x m, where n - number of users, m - number of all interests. In our case it will be 15 x 36


```python
def make_user_interest_vector(user_interests):
    return [1 if interest in user_interests else 0 for interest in unique_interests]
```


```python
user_interest_matrix = list(map(make_user_interest_vector, users_interests))
```


```python
print(user_interest_matrix[0]) # first user interests
```

    [1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


After matrix is completed, we can go ahead and compute cosine similarity against each user


```python
user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
    for interest_vector_j in user_interest_matrix]
    for interest_vector_i in user_interest_matrix]
```


```python
def most_similar_users_to(user_id):
    pairs = [(other_user_id, similarity) # find other
            for other_user_id, similarity in # users with
            enumerate(user_similarities[user_id]) # nonzero
            if user_id != other_user_id and similarity > 0] # similarity
    
    return sorted(pairs, key=lambda similarity: similarity[1], reverse=True)
```


```python
most_similar_users_to(0) # user with id 0 most similar to user 9, 1,...,5
```




    [(9, 0.5669467095138409),
     (1, 0.3380617018914066),
     (8, 0.1889822365046136),
     (13, 0.1690308509457033),
     (5, 0.1543033499620919)]




```python
def user_based_suggestions(user_id, include_current_interests=False):
    
    # sum up the similarities
    suggestions = defaultdict(float)
    for other_user_id, similarity in most_similar_users_to(user_id):
        for interest in users_interests[other_user_id]:
            suggestions[interest] += similarity
    
    # convert them to a sorted list
    suggestions = sorted(suggestions.items(), key=lambda weight: weight[1], reverse=True)
    
    # and (maybe) exclude already-interests
    if include_current_interests:
        return suggestions
    else:
        return [(suggestion, weight)
                for suggestion, weight in suggestions
                if suggestion not in users_interests[user_id]]
```

Here we essentially filter out interests of most similar users without including interests of a user


```python
user_based_suggestions(0, include_current_interests=False) # user 0 was suggested following interests
```




    [('MapReduce', 0.5669467095138409),
     ('MongoDB', 0.50709255283711),
     ('Postgres', 0.50709255283711),
     ('NoSQL', 0.3380617018914066),
     ('neural networks', 0.1889822365046136),
     ('deep learning', 0.1889822365046136),
     ('artificial intelligence', 0.1889822365046136),
     ('databases', 0.1690308509457033),
     ('MySQL', 0.1690308509457033),
     ('Python', 0.1543033499620919),
     ('R', 0.1543033499620919),
     ('C++', 0.1543033499620919),
     ('Haskell', 0.1543033499620919),
     ('programming languages', 0.1543033499620919)]



## 2. Item-Based Collaborative Filtering


```python
interest_user_matrix = [[user_interest_vector[j] for user_interest_vector in user_interest_matrix]
    for j, _ in enumerate(unique_interests)]
```


```python
interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
        for user_vector_j in interest_user_matrix]
        for user_vector_i in interest_user_matrix]
```


```python
def most_similar_interests_to(interest: str) -> list:
    interest_id = unique_interests.index('Big Data')
    similarities = interest_similarities[interest_id]
    pairs = [(unique_interests[other_interest_id], similarity)
            for other_interest_id, similarity in enumerate(similarities)
            if interest_id != other_interest_id and similarity > 0]
    return sorted(pairs, key=lambda similarity: similarity[1], reverse=True)
```


```python
most_similar_interests_to('Big Data')
```




    [('Hadoop', 0.8164965809277261),
     ('Java', 0.6666666666666666),
     ('MapReduce', 0.5773502691896258),
     ('Spark', 0.5773502691896258),
     ('Storm', 0.5773502691896258),
     ('Cassandra', 0.4082482904638631),
     ('artificial intelligence', 0.4082482904638631),
     ('deep learning', 0.4082482904638631),
     ('neural networks', 0.4082482904638631),
     ('HBase', 0.3333333333333333)]




```python
def item_based_suggestions(user_id: int) -> list:
    # add up the similar interests
    suggestions = defaultdict(float)
    user_interest_vector = user_interest_matrix[user_id]
    for interest_id, is_interested in enumerate(user_interest_vector):
        if is_interested == 1:
            similar_interests = most_similar_interests_to(interest_id)
            for interest, similarity in similar_interests:
                suggestions[interest] += similarity
    
    # sort them by weight
    suggestions = sorted(suggestions.items(), key=lambda similarity: similarity[1], reverse=True)
    return suggestions
```


```python
item_based_suggestions(0)
```




    [('Hadoop', 5.715476066494083),
     ('Java', 4.666666666666666),
     ('MapReduce', 4.041451884327381),
     ('Spark', 4.041451884327381),
     ('Storm', 4.041451884327381),
     ('Cassandra', 2.8577380332470415),
     ('artificial intelligence', 2.8577380332470415),
     ('deep learning', 2.8577380332470415),
     ('neural networks', 2.8577380332470415),
     ('HBase', 2.333333333333333)]



## Conclusion

Both approaches returned identical results
