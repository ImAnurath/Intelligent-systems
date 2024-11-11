#LAB-1
import random
weight1 = random.uniform(-1, 1)
weight2 = random.uniform(-1, 1)
bias = random.uniform(-1, 1)
learning_rate = 0.1
iteration = 0
max_iteration = 1000

def update_parameters(w1, w2, b, lr, e, x1, x2):
    w1 = w1 + lr * e * x1
    w2 = w2 + lr * e * x2
    b  = b  + lr * e
    return w1, w2, b
def calculate_error(data):
    error = []
    for d in data:
        actual_output = 1 if ((weight1 * d[0]) + (weight2 * d[1]) + bias) > 0 else -1
        error.append(d[2] - actual_output)
    
    return error

datas = [
    [0.21835,  0.81884,  1],
    [0.14115,  0.83535,  1],
    [0.37022,  0.8111,   1],
    [0.31565,  0.83101,  1],
    [0.36484,  0.8518,   1],
    [0.46111,  0.82518,  1],
    [0.55223,  0.83449,  1],
    [0.16975,  0.84049,  1],
    [0.49187,  0.80889,  1],
    [0.14913,  0.77104, -1],
    [0.18474,  0.6279,  -1],
    [0.08838,  0.62068, -1],
    [0.098166, 0.79092, -1]
]

while iteration < max_iteration:
    errors = calculate_error(datas)
    if all(e == 0 for e in errors):
        print("Converged.")
        break
    
    for index, data in enumerate(datas):
       weight1, weight2, bias = update_parameters(weight1, weight2, bias, learning_rate, errors[index], data[0], data[1])
       
    iteration += 1
    print(f"Iteration: {iteration}  Weight1: {weight1}, Weight2: {weight2}, Bias: {bias}")
#AV