# einfache version eines neuronalen netzes
import numpy as np
import math

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

X = np.array([[0,0,1],
              [0,1,0],
              [1,0,0],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])

nn = NeuralNetwork(X,y)

loss_values = []


for i in range(2000):
    nn.feedforward()
    nn.backprop()
    loss = compute_loss(nn.output, y)
    loss_values.append(loss)

print(nn.output)
print(f" final loss : {loss}")
















# brauchbare version eines neuronalen netzes


##### Eingabe #####
num_samples = 500
pi = np.pi
aussen = 1
innen = 0.7
training_anzahl = 10
points = np.zeros((num_samples,2,training_anzahl))

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],2*training_anzahl) 
        self.weights2   = np.random.rand(2*training_anzahl,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        
# simulierte kreisringe
for i in range(training_anzahl):
    phi = np.random.uniform(0, 2*pi, num_samples)
    r = np.random.uniform(innen, aussen, num_samples)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    points[:,:,i] = np.vstack((x,y)).T

# punkte der kreisringe in eine matrix
X1 = np.zeros((training_anzahl,1000))
for i in range(training_anzahl):
    X1[i] = points[:,:,i].reshape(1,1000)

# simuliere offene kreisringe
for i in range(training_anzahl):
    phi = np.random.uniform(0, 1.5*pi, num_samples)
    r = np.random.uniform(innen, aussen, num_samples)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    points[:,:,i] = np.vstack((x,y)).T

# punkte der offenen kreisringe in eine Matrix
X2 = np.zeros((training_anzahl,1000))
for i in range(training_anzahl):
    X2[i] = points[:,:,i].reshape(1,1000)

# die inputmatrix für das NN
X = np.zeros((2*training_anzahl,1000))
for i in range(10):
    print(2*i)
    X[2*i] = X1[i]
    X[2*i+1] = X2[i] 

# der inputvektor für das NN
y = np.array([[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0],[1],[0]])

nn = NeuralNetwork(X,y)

loss_values = []

for i in range(200000):
    nn.feedforward()
    nn.backprop()
    loss = compute_loss(nn.output, y)
    loss_values.append(loss)

print(nn.output)
print(f" final loss : {loss}")

plt.plot(loss_values)






# test
phi = np.random.uniform(0, 1*pi, num_samples)
r = np.random.uniform(innen, aussen, num_samples)

x_test = r * np.cos(phi)
y_test = r * np.sin(phi)
points = np.vstack((x_test,y_test)).T
punkte = points.reshape(1,1000)
l1 = sigmoid(np.dot(punkte, nn.weights1))
output = sigmoid(np.dot(l1, nn.weights2))

print(output)
