import random

def initialize_weights():
    return random.uniform(-0.5, 0.5)


def exp_approx(x, terms=10):
    result = 1
    numerator = 1
    denominator = 1
    for n in range(1, terms):
        numerator *= x
        denominator *= n
        result += numerator / denominator
    return result

def tanh(x):
    e_x = exp_approx(x)
    e_neg_x = exp_approx(-x)
    return (e_x - e_neg_x) / (e_x + e_neg_x)


def squared_error(target, output):
    return 0.5 * (target - output) ** 2


i1, i2 = 0.05, 0.10  
b1, b2 = 0.5, 0.7  

w1, w2, w3, w4 = initialize_weights(), initialize_weights(),initialize_weights(), initialize_weights()
w5, w6, w7, w8 = initialize_weights(), initialize_weights(),initialize_weights(), initialize_weights()

h1_input = (i1 * w1) + (i2 * w3) + b1
h2_input = (i1 * w2) + (i2 * w4) + b1

h1_output = tanh(h1_input)
h2_output = tanh(h2_input)

o1_input = (h1_output * w5) + (h2_output * w7) + b2
o2_input = (h1_output * w6) + (h2_output * w8) + b2

o1_output = tanh(o1_input)
o2_output = tanh(o2_input)

target_o1, target_o2 = 0.01, 0.99 

error_o1 = squared_error(target_o1, o1_output)
error_o2 = squared_error(target_o2, o2_output)

total_error = error_o1 + error_o2

print("Output of the neural network:")
print("o1 =", o1_output)
print("o2 =", o2_output)
print("\nTotal Error =", total_error)

