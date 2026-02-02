import tensorflow as tf

A = tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

# Matrix addition
print("Addition:\n", tf.add(A, B))

# Matrix multiplication
print("Multiplication:\n", tf.matmul(A, B))

# Transpose
print("Transpose:\n", tf.transpose(A))

# Determinant
print("Determinant:", tf.linalg.det(A))

# Inverse
print("Inverse:\n", tf.linalg.inv(A))
