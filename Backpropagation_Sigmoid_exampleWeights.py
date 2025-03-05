def exp_approx(x, terms=10):
    result = 1
    numerator = 1
    denominator = 1
    for n in range(1, terms):
        numerator *= x
        denominator *= n
        result += numerator / denominator
    return result

def sigmoid(x):
    return 1 / (1 + exp_approx(-x))

def sigmoid_derivative(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

def squared_error(target, output):
    return 0.5 * (target - output) ** 2

i1, i2 = 0.05, 0.10  
b1, b2 = 0.35, 0.60  
learning_rate = 0.5

w1, w2, w3, w4 = 0.15, 0.20, 0.25, 0.30
w5, w6, w7, w8 = 0.40, 0.45, 0.50, 0.55

target_o1, target_o2 = 0.01, 0.99 

print("Initial Weights:")
print(f"w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}, w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}")

for epoch in range(5):
    # Forward Pass
    h1_input = (i1 * w1) + (i2 * w3) + b1
    h2_input = (i1 * w2) + (i2 * w4) + b1
    
    h1_output = sigmoid(h1_input)
    h2_output = sigmoid(h2_input)
    
    o1_input = (h1_output * w5) + (h2_output * w7) + b2
    o2_input = (h1_output * w6) + (h2_output * w8) + b2
    
    o1_output = sigmoid(o1_input)
    o2_output = sigmoid(o2_input)
    
    error_o1 = squared_error(target_o1, o1_output)
    error_o2 = squared_error(target_o2, o2_output)
    total_error = error_o1 + error_o2
    
    print(f"Epoch {epoch} Forward Pass:")
    print(f"o1 output: {o1_output}, o2 output: {o2_output}, Total Error: {total_error}")
    
    # Backward Pass
    d_error_o1 = o1_output - target_o1
    d_error_o2 = o2_output - target_o2
    
    d_o1 = d_error_o1 * sigmoid_derivative(o1_input)
    d_o2 = d_error_o2 * sigmoid_derivative(o2_input)
    
    d_w5 = d_o1 * h1_output
    d_w6 = d_o2 * h1_output
    d_w7 = d_o1 * h2_output
    d_w8 = d_o2 * h2_output
    
    d_h1 = (d_o1 * w5) + (d_o2 * w6)
    d_h2 = (d_o1 * w7) + (d_o2 * w8)
    
    d_h1_input = d_h1 * sigmoid_derivative(h1_input)
    d_h2_input = d_h2 * sigmoid_derivative(h2_input)
    
    d_w1 = d_h1_input * i1
    d_w2 = d_h2_input * i1
    d_w3 = d_h1_input * i2
    d_w4 = d_h2_input * i2
    
    # Update Weights
    w1 -= learning_rate * d_w1
    w2 -= learning_rate * d_w2
    w3 -= learning_rate * d_w3
    w4 -= learning_rate * d_w4
    w5 -= learning_rate * d_w5
    w6 -= learning_rate * d_w6
    w7 -= learning_rate * d_w7
    w8 -= learning_rate * d_w8
    
    print(f"Epoch {epoch} Backpropagation:")
    print(f"Updated Weights: w1: {w1}, w2: {w2}, w3: {w3}, w4: {w4}, w5: {w5}, w6: {w6}, w7: {w7}, w8: {w8}\n")
