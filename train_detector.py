import numpy as np
from sklearn import neighbors

from config import ATTACK_DICT, SAMPLE_LOCATION

N = 100
knn_model = neighbors.KNeighborsClassifier(n_neighbors=N)
print('knn based detector, k value is:', N)

# train
benign_cost = np.load(SAMPLE_LOCATION / 'benign_cost.npy')
attack_cost = {}
for attack_type in ATTACK_DICT.keys():
    attack_cost[attack_type] = np.load(SAMPLE_LOCATION / f'{attack_type}_cost.npy')
costs_train = np.concatenate([benign_cost, *attack_cost.values()]).reshape(-1, 1)
labels_train = np.concatenate([np.zeros(len(benign_cost)), np.ones(costs_train.shape[0] - len(benign_cost))])
knn_model.fit(costs_train, labels_train)
print('training acc:', knn_model.score(costs_train, labels_train))

# Test
result = knn_model.predict(benign_cost.reshape(-1, 1))
acc = 1 - sum(result) / len(result)
print('benign acc:', acc)
for atk_type, costs in attack_cost.items():
    result = knn_model.predict(costs.reshape(-1, 1))
    acc = sum(result) / len(result)
    print(f'{atk_type} acc:', acc)
