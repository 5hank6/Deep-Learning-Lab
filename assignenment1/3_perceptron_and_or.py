import numpy as np

def perceptron(x, w, b):
    return 1 if np.dot(x, w) + b >= 0 else 0

# AND gate
print("AND Gate")
w_and = np.array([1, 1])
b_and = -1.5
for x in [(0,0), (0,1), (1,0), (1,1)]:
    print(x, perceptron(x, w_and, b_and))

# OR gate
print("\nOR Gate")
w_or = np.array([1, 1])
b_or = -0.5
for x in [(0,0), (0,1), (1,0), (1,1)]:
    print(x, perceptron(x, w_or, b_or))
