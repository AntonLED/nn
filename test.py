# import tensorflow as tf
# from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt 


# x_train = np.loadtxt("./datasets/data/in", dtype=np.float64)
# print(x_train)

# y_train = np.loadtxt("./datasets/data/out")
# print(y_train)

x_train = np.loadtxt("./datasets/data/in", dtype=np.float64)
y_train = np.loadtxt("./datasets/data/out", dtype=np.float64)
x_train_norm = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
y_train_norm = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

fig, ax = plt.subplots(2)
# fig.dpi = 300
ax[0].hist(x_train)
ax[1].hist(x_train_norm)
plt.show()


fig, ax = plt.subplots(2)
# fig.dpi = 300
ax[0].hist(y_train)
ax[1].hist(y_train_norm)
plt.show()

print(np.median(x_train))
print(np.median(x_train_norm))
# x_train_norm = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))
# y_train_norm = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))

# print(x_train_norm[0])

# print(x_train_norm * (np.max(x_train) - np.min(x_train)) + np.min(x_train) - x_train)
# print(y_train_norm * (np.max(y_train) - np.min(y_train)) + np.min(y_train) - y_train)



