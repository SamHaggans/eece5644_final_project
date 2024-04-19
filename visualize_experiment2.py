# Box plot generation helper from data from experiment2.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

nn_predict = [9931.4,  8464.8, 986.4, 873.1, 2595.6, 8699.3,  967.3]
real_means = [9930.0,7291.0, 1016.0,681.0, 2258.0, 8040.0, 790.0]
b_predict = [7523.38, 7632.89, 1686.32,  1663.32, 2990.47, 7547.07, 1658.48]
b_stdev = [3793.87,3867.95, 755.81, 746.01, 1397.35, 3811.24, 743.68]


fig, axs = plt.subplots(1, 7, sharey=True, figsize=(20, 2))

for i, _ in enumerate(nn_predict):

    data = np.random.normal(loc=b_predict[i], scale=b_stdev[i], size=(10000,))

    # Create a box and whisker plot
    axs[i].boxplot(data, showfliers=False)
    axs[i].scatter([1], [real_means[i]], marker="x", color="blue", label="True Label")
    axs[i].scatter([1], [nn_predict[i]], marker="o", color="red", label="Conventional NN Output")

    # Add title and labels
    axs[i].set_ylabel('Diamond Price')

    # Show plot
    
axs[0].legend(loc="lower right")
plt.show()