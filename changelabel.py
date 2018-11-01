import numpy as np

input_file = 'multivar_data_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
data[:, -1] = [0 if x < 2 else 1 for x in data[:, -1]]
np.savetxt('new_multivar_data_nb.txt', data, delimiter=',', fmt=['%.2f', '%.2f', '%d'])
