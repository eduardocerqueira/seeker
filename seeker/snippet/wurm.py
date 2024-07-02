#date: 2024-07-02T16:54:33Z
#url: https://api.github.com/gists/15c3876f5984f5e71b4548ef79da3b3c
#owner: https://api.github.com/users/MightyPiggie

import matplotlib.pyplot as plt
import numpy as np

# Formula constants
valueAtOneHundred = 0.9
valueAtZero = 3.74
negativeDecayRate = 5
positiveDecayRate = 3


# Formula to calculate the multiplier at power level
def calculateMultiplier(power):
    return valueAtOneHundred*pow(valueAtZero/valueAtOneHundred, (2-pow(100/(100+max(-99,power)), negativeDecayRate))*pow((100-power)*0.01, positiveDecayRate))

# create array with all possible power levels
x = np.arange(-100,101,1)
y_new = np.copy(x)
y_old = np.copy(x)

# Apply formula with original constants on curve
applyall = np.vectorize(calculateMultiplier)
y_old = applyall(y_old)

# Changed constants
valueAtOneHundred = 1
valueAtZero = 4
negativeDecayRate = 5
positiveDecayRate = 5

# Apply formula with changed constants on curve
y_new = applyall(y_new)

# Create plot
new = plt.plot(x, y_new)
old = plt.plot(x, y_old)
plt.xlabel('power')
plt.ylabel('multiplier')
plt.legend(["new", "old"])
plt.grid()

# plt.plot([0, 100], [1, 1], 'k-', lw=2)

plt.show()