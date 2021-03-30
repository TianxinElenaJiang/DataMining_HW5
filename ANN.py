import numpy as np

class NeuralNetwork(object):
    def __init__(self,
                 layers,
                 activation='ANN',
                 input_notes=25,
                 output_nodes=10):

        self.weight = []
        self.previous_delta = []
        layers.insert(0, input_notes)
        layers.insert(len(layers), output_nodes)

        for i in range(1, len(layers) - 1):
            self.weight.append(
                (2 * np.random.random(
                    (layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
            self.previous_delta.append(
                (np.zeros((layers[i - 1] + 1, layers[i] + 1))))
        i = i + 1
        self.weight.append(
            (2 * np.random.random((layers[i - 1] + 1, layers[i])) - 1) * 0.25)
        self.previous_delta.append((np.zeros((layers[i - 1] + 1, layers[i]))))

    def fit(self, x_arr, y_arr,
            learning_rate=0.001, num_epochs=400000, momentum=0.0, lmbda=0.1):
        n = len(x_arr)
        x_arr = np.atleast_2d(x_arr)
        temp = np.ones([x_arr.shape[0], x_arr.shape[1] + 1])
        temp[:, 0:-1] = x_arr
        x_arr = temp
        y_arr = np.array(y_arr)

        for k in range(num_epochs):
            i = np.random.randint(x_arr.shape[0])
            a = [x_arr[i]]
            for l in range(0, len(self.weight)):
                a.append(self.activation(np.dot(a[l], self.weight[l])))
            error = y_arr[i] - a[-1]
            deltas = [error * self.activation_deriv(a[-1])]
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(
                    self.weight[l].T) * self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weight)):
                layer = np.atleast_2d(a[i])
                current_delta = np.atleast_2d(deltas[i])
                delta = (learning_rate * layer.T.dot(current_delta) +
                         momentum * self.previous_delta[i])
                self.weight[i] = ((1 - (learning_rate * (lmbda / n))) *
                                   self.weight[i] + delta)
                self.previous_delta[i] = delta

    def predict(self, x_test):
        x_test = np.array(x_test)
        temp = np.ones(x_test.shape[0] + 1)
        temp[0:-1] = x_test
        a = temp
        for l in range(0, len(self.weight)):
            a = self.activation(np.dot(a, self.weight[l]))
        return a

