import numpy as np
from scipy import optimize

def f(x):
    return (x+3)**2 + 5

minimum = optimize.fmin(f, 1)

print(minimum[0])
