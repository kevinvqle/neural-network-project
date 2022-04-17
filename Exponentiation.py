import math 
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

E = math.e
exp_values = np.exp(layer_outputs)

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_base)
print(sum(norm_values))