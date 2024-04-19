# Data generated from the output of an uncertainty threshold parameter sweep in experiment1.py
data = [((89.6484375, 100.0), 0.0), ((89.453125, 100.0), 0.05), ((89.35546875, 100.0), 0.1), ((91.015625, 100.0), 0.15000000000000002), ((91.30859375, 100.0), 0.2), ((89.59764474975465, 99.51171875), 0.25), ((90.08016032064128, 97.4609375), 0.30000000000000004), ((91.74311926605505, 95.80078125), 0.35000000000000003), ((92.36326109391125, 94.62890625), 0.4), ((94.04888416578109, 91.89453125), 0.45), ((96.74887892376681, 87.109375), 0.5), ((95.97156398104265, 82.421875), 0.55), ((97.9064039408867, 79.296875), 0.6000000000000001), ((98.39357429718876, 72.94921875), 0.65), ((99.28263988522238, 68.06640625), 0.7000000000000001), ((99.69879518072288, 64.84375), 0.75), ((98.7719298245614, 55.6640625), 0.8), ((99.76689976689977, 41.89453125), 0.8500000000000001), ((99.69325153374233, 31.8359375), 0.9), ((100.0, 16.2109375), 0.9500000000000001)]

import numpy as np
import matplotlib.pyplot as plt

correct_rate = [dat[0][0] / 100 for dat in data]
prediction_rate = [dat[0][1] / 100 for dat in data]

plt.figure(figsize=(8, 8))
plt.plot(prediction_rate, correct_rate, color="darkorange", lw=2, label="Curve")
plt.ylabel("Correct Labeling Rate")
plt.xlabel("Data Labeling Rate")
plt.title("Data Labeling Characteristics vs Confidence Threshold")
plt.legend(loc="lower right")
plt.show()