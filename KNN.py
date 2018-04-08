import numpy as np
from collections import Counter
import warnings
import random
import pandas as pd


def k_nearest_neighbors(data, predict, k=5):
    if k > sum(len(v) for v in data.values()):
        warnings.warn('K is set to a value more than total data points!')
    distances = []

    for label in data:
        for features in data[label]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))

            distances.append([euclidean_distance, label])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_results = Counter(votes).most_common(1)[0][0]

    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_results, confidence


df = pd.read_csv('breast-cancer-wisconsin.data.txt')

df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

test_size = 0.2
train_data = full_data[:-int(test_size * len(full_data))]
test_data = full_data[-int(test_size * len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for label in test_set:
    for predict in test_set[label]:
        vote, confidence = k_nearest_neighbors(train_set, predict, k=5)

        if vote == label:
            correct += 1
        else:
            print(confidence)

        total += 1

print('Accuracy:', correct/total)
