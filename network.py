"""
Main class of the Neural Network. 
"""

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input):
        samples = len(input)
        result = []

        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        return result
    
    def fit(self, x_train, y_train, epochs, learn_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0

            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learn_rate)
        
            err /= samples
            print(f"epoch {i+1}   error={err:.6f}")