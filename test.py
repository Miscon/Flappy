# from flappy_agent import FlappyAgent

# agent = FlappyAgent()

import numpy as np
import matplotlib.pyplot as plt
import math


# step = math.floor(float(100)/3)
# print(step)
# bins = np.concatenate(([-300], np.arange(1, 101, step), [300]), axis=0)
# print(bins)
# values = np.digitize([-240, 0, 240], bins)
# print(values)

# diff = 157  # min and max possible values
# step = math.floor(float(diff)/2) # split in 4 bins
# bins = np.arange(-diff, diff, step)
# print(step)
# print(bins)
# step = int((203+230)/15) + 1
# print(step)
# bins = np.arange(-202, 230, step)

# binone = np.digitize([50], bins)
# print(type(binone[0]))

# bins = [x-1 for x in np.geomspace(1, 66, num=4)]
# print(bins)


# print(np.linspace(13, 76, 11))


# print([a-9 for a in np.geomspace(1, 20, 5)])



print(np.linspace(13, 76, 4))
print([b-1 for b in np.geomspace(1, 66, num=x))


# value = np.digitize([1, 2, 4, 5], [2, 4, 6])
# print(value)

# def plot(i, ax):
#     y = np.random.random()
#     ax.scatter(i, y)


# f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# for i in range(100):
#     ax1.cla()
#     plot(i, ax1)
#     plot(i, ax2)

#     plt.pause(0.00000001)
#     import time
#     time.sleep(1)




# import numpy as np

# # weights = np.array([0] * 4)
# # features = np.array([256.,  10., 309., 174.])

# # for i in range(10):
# #     G = -5

# #     print('---------------------------------------------------\n---------------------------------------------------')
# #     print("np.dot(weights, features)==== ", np.dot(weights, features))
# #     print("(np.dot(weights, features) - G)==== ", (np.dot(weights, features) - G))
# #     print("(np.dot(weights, features) - G) * feature====s ", (np.dot(weights, features) - G) * features)
# #     print("0.1 * (np.dot(weights, features) - G) * features==== ", 0.1 * (np.dot(np.transpose(weights), features) - G) * features)
# #     print("weights - 0.1 * (np.dot(weights, features) - G) * features==== ", weights - 0.1 * (np.dot(weights, features) - G) * features)

# #     weights = weights - 0.1 * (np.dot(weights, features) - G) * features
# #     print(weights)


# # print('===================================================\n===================================================')




# gamma = 0.8

# G = 1

# for i in range(40):
#     G = 0.01 + gamma * G

# G = 1 + gamma * G

# print(G)


