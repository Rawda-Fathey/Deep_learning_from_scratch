import random

def exp(x):
    result = 1.0
    term = 1.0
    for n in range(1, 20):  
        term *= x / n
        result += term
    return result

def tanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))

def tanh_deriv(x):
    t = tanh(x)
    return 1 - t * t

def softmax(x):
    exp_x = [exp(xi) for xi in x]
    sum_exp_x = sum(exp_x)
    return [ex / sum_exp_x for ex in exp_x]

def cross_entropy_loss(y_true, y_pred):
    return -sum([t * (p + 1e-10) for t, p in zip(y_true, y_pred)])

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.W_xh = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.W_hh = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.W_hy = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
        self.b_h = [0.0] * hidden_size
        self.b_y = [0.0] * output_size

    def forward(self, inputs):
        h = [0.0] * self.hidden_size
        self.h_states = [h[:]] 
        self.outputs = []
        
        for x in inputs:
            h_new = []
            for i in range(self.hidden_size):
                z = sum(self.W_xh[i][j] * x[j] for j in range(self.input_size))
                z += sum(self.W_hh[i][j] * h[j] for j in range(self.hidden_size))
                z += self.b_h[i]
                h_new.append(tanh(z))
            h = h_new
            self.h_states.append(h[:])
            
            y = []
            for i in range(self.output_size):
                z = sum(self.W_hy[i][j] * h[j] for j in range(self.hidden_size))
                z += self.b_y[i]
                y.append(z)
            y = softmax(y)
            self.outputs.append(y)
        
        return self.outputs[-1], h

    def backward(self, inputs, target, learning_rate=0.01):
        dW_xh = [[0.0] * self.input_size for _ in range(self.hidden_size)]
        dW_hh = [[0.0] * self.hidden_size for _ in range(self.hidden_size)]
        dW_hy = [[0.0] * self.hidden_size for _ in range(self.output_size)]
        db_h = [0.0] * self.hidden_size
        db_y = [0.0] * self.output_size
        
        y_pred = self.outputs[-1]
        dy = [y_pred[i] - target[i] for i in range(self.output_size)]
        
        dh_next = [0.0] * self.hidden_size
        for t in range(len(inputs) - 1, -1, -1):
            h = self.h_states[t + 1]
            h_prev = self.h_states[t]
            x = inputs[t]
            
            if t == len(inputs) - 1:
                for i in range(self.output_size):
                    for j in range(self.hidden_size):
                        dW_hy[i][j] += dy[i] * h[j]
                    db_y[i] += dy[i]
            
            dh = [0.0] * self.hidden_size
            if t == len(inputs) - 1:
                for j in range(self.hidden_size):
                    dh[j] = sum(dy[i] * self.W_hy[i][j] for i in range(self.output_size))
            
            dh_raw = [dh[j] * tanh_deriv(sum(self.W_xh[j][k] * x[k] for k in range(self.input_size)) +
                                     sum(self.W_hh[j][k] * h_prev[k] for k in range(self.hidden_size)) +
                                     self.b_h[j]) for j in range(self.hidden_size)]
            
            for j in range(self.hidden_size):
                for k in range(self.input_size):
                    dW_xh[j][k] += dh_raw[j] * x[k]
                db_h[j] += dh_raw[j]
            
            for j in range(self.hidden_size):
                for k in range(self.hidden_size):
                    dW_hh[j][k] += dh_raw[j] * h_prev[k]
            
            dh_next = [sum(dh_raw[j] * self.W_hh[j][k] for j in range(self.hidden_size)) for k in range(self.hidden_size)]
        
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                self.W_xh[i][j] -= learning_rate * dW_xh[i][j]
            for j in range(self.hidden_size):
                self.W_hh[i][j] -= learning_rate * dW_hh[i][j]
            self.b_h[i] -= learning_rate * db_h[i]
        
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                self.W_hy[i][j] -= learning_rate * dW_hy[i][j]
            self.b_y[i] -= learning_rate * db_y[i]

inputs = [
    [1, 0, 0, 0],  # colors
    [0, 1, 0, 0],  # blend
    [0, 0, 1, 0]   # in
]
target = [0, 0, 0, 1]  # twilight

rnn = SimpleRNN(input_size=4, hidden_size=3, output_size=4)
for epoch in range(1000):
    output, _ = rnn.forward(inputs)
    loss = cross_entropy_loss(target, output)
    rnn.backward(inputs, target, learning_rate=0.01)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Predicted: {output}")

output, _ = rnn.forward(inputs)
print("\nFinal prediction:", output)
print("Target (twilight):", target)