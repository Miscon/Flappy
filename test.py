# from flappy_agent import FlappyAgent

# agent = FlappyAgent()

import numpy as np
import matplotlib.pyplot as plt

step = int((203+230)/15) + 1
print(step)
bins = np.arange(-202, 230, step)

binone = np.digitize([50], bins)
print(type(binone[0]))





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


