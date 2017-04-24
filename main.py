import pandas as pd
import numpy as np

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


def split_data(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice()