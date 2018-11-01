import numpy as np
from scipy.stats import stats
from sklearn import preprocessing
input_data = np.array([[5, -2, 3], [-1, 7, -6], [3, 0, 2], [7, -9, -4]])
print(input_data)

data_binarized = preprocessing.Binarizer(threshold=2.2).transform(input_data)
print("\nBinarized data:\n", data_binarized)

print("axis=0")
print("Mean =", input_data.mean(axis=0))
print("variance =", input_data.var(axis=0))
print("Std deviation =", input_data.std(axis=0))
print("axis=1")
print("Mean =", input_data.mean(axis=1))
print("variance =", input_data.var(axis=1))
print("Std deviation =", input_data.std(axis=1))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("variance =", input_data.var(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

minmax_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = minmax_scaler.fit_transform(input_data)
print("\nMin-max scaled data:\n", data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nL1 normalized data:\n", data_normalized_l1)
print("\nL2 normalized data:\n", data_normalized_l2)

homework = np.array([23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61])
homework2 = np.array([9.5, 26.5, 7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2, 34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2
                         , 35.7])
print(homework.mean(), homework2.mean())
print(homework.std(), homework2.std())
print(stats.zscore(homework))
print(stats.zscore(homework2))
print(np.corrcoef(homework, homework2)[0, 1])
