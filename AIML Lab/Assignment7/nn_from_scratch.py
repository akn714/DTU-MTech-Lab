"""
Artificial Neural Network (ANN) implementation from scratch (without backpropogation)
"""

import math

def relu(x):
    return max(0, x)

def softmax(x):
    exp_x = [math.exp(i) for i in x]
    sum_exp_x = sum(exp_x)
    return [i/sum_exp_x for i in exp_x]

def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))

def log_loss(y_true, y_pred):
    return -sum([y_true[i] * math.log(y_pred[i]) for i in range(len(y_true))])

def matmult(mat1, mat2):
    z = [
        [0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))
    ]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                z[i][j] += mat1[i][k] * mat2[k][j]

    return z

def transpose(mat):
    out = [[mat[i][j] for i in range(len(mat))] for j in range(len(mat[0]))]
    return out

def feedforward(inp):
    # input layer
    inp = inp

    # layer 1 (6 neurons)
    w1 = [[x*y*0.01 for x in range(1, 5)] for y in range(1, 7)] # 4x6 weight matrix
    z1 = matmult(inp, transpose(w1))
    a1 = [[relu(i) for i in z1[0]]] # 1x6 output matrix

    # layer 2 (5 neurons)
    w2 = [[x*y*0.01 for x in range(1, 7)] for y in range(1, 6)] # 6x5 weight matrix
    z2 = matmult(a1, transpose(w2))
    a2 = [softmax(z2[0])] # 1x5 output matrix

    # layer 3 (4 neurons)
    w3 = [[x*y*0.01 for x in range(1, 6)] for y in range(1, 5)] # 5x4 weight matrix
    z3 = matmult(a2, transpose(w3))
    a3 = [[tanh(i) for i in z3[0]]] # 1x4 output matrix

    # layer 4 (3 neurons)
    w4 = [[x*y*0.01 for x in range(1, 5)] for y in range(1, 4)] # 4x3 weight matrix
    z4 = matmult(a3, transpose(w4))
    a4 = [[relu(i) for i in z4[0]]] # 1x3 output matrix

    # output layer (3 neurons)
    w5 = [[x*y*0.01 for x in range(1, 4)] for y in range(1, 4)] # 4x3 weight matrix
    z5 = matmult(a4, transpose(w5))
    a5 = [softmax(z5[0])] # 1x3 output matrix

    return a5


INPUT = [[1,2,1,3]]
OUTPUT = [0,1,0]

PREDICTED = feedforward(INPUT)[0]
LOSS = log_loss(OUTPUT, PREDICTED)

print("Predicted output: ", PREDICTED)
print("Loss: ", LOSS)
