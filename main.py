import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', sep = '\t', names = names)
# print(df.head())
users = df.user_id.unique().shape[0]
items = df.item_id.unique().shape[0]

ratings = np.zeros((users, items))

for row in df.itertuples():
    ratings[row[1]-1, row[2]-1] = row[3]

# Sparsity Check
sparsity = float(len(ratings.nonzero()[0]))
sparsity /= (ratings.shape[0]*ratings.shape[1])
# print(100*sparsity) # These many percentage of ratings have a value > 0

# Test and train, 10 ratings from every user will be stored in test from train data
def split_data(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in range(ratings.shape[0]):
        # selects 10 random user inputs for test
        test_ratings = np.random.choice(ratings[user,:].nonzero()[0], size = 10, replace = False)
        # make test users value 0 in train to avoid bias
        train[user, test_ratings] = 0
        test[user, test_ratings] = ratings[user, test_ratings]
    assert(np.all((train*test) == 0))
    return train, test

train, test = split_data(ratings)


# Cosine similarity using numpy
def fast_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon is a small number to handle dividing by zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return sim/norms/norms.T

user_similarity = fast_similarity(train, kind='user')
# item_similarity = fast_similarity(train, kind='item')

#FAST METHOD USING NUMPY
def predict_fast_simple(ratings, similarity, kind= 'user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis = 1)]).T
    elif kind == 'item':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis = 1)])


def get_mse(pred, actual):
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)


user_prediction = predict_fast_simple(train, user_similarity, kind= 'user')
print('User-based CF MSE: ' + str(get_mse(user_prediction, test)))

# TOP K COLLABORATIVE FILTERING, to improve prediction
def predict_topk(ratings, similarity, kind='user', k=40):
    pred = np.zeros(ratings.shape)
    if kind == 'user':
        for i in range(ratings.shape[0]):
            top_k_users = [np.argsort(similarity[:, i])[: -k-1:-1]]
            for j in range(ratings.shape[1]):
                pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users])
                pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
    if kind == 'item':
        for j in range(ratings.shape[1]):
            top_k_items = [np.argsort(similarity[:, j])[:-k-1:-1]]
            for i in range(ratings.shape[0]):
                pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T)
                pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))
    return pred

pred = predict_topk(train, user_similarity, kind='user', k=40)
print('User based CF MSE: ' + str(get_mse(pred, test)))