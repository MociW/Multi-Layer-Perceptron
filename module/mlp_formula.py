import numpy as np   # for numerical calculation
import random        # for making random number
import matplotlib.pyplot as plt

# Define MLP Class Object


class MultiLayerPerceptron:
    def __init__(self, params=None):
        # Defaulf MLP Layer if not specify
        if (params == None):
            self.inputLayer = 4                        # Input Layer
            self.hiddenLayer = 5                       # Hidden Layer
            self.outputLayer = 3                       # Outpuy Layer
            self.learningRate = 0.005                  # Learning rate
            self.max_epochs = 600                      # Epochs
            self.BiasHiddenValue = -1                  # Bias HiddenLayer
            self.BiasOutputValue = -1                  # Bias OutputLayer
            self.activation = self.activation['sigmoid']  # Activation function
            self.deriv = self.deriv['sigmoid']
        else:
            self.inputLayer = params['InputLayer']
            self.hiddenLayer = params['HiddenLayer']
            self.OutputLayer = params['OutputLayer']
            self.learningRate = params['LearningRate']
            self.max_epochs = params['Epochs']
            self.BiasHiddenValue = params['BiasHiddenValue']
            self.BiasOutputValue = params['BiasOutputValue']
            self.activation = self.activation[params['ActivationFunction']]
            self.deriv = self.deriv[params['ActivationFunction']]

        # Initialize Weight and Bias value
        self.WEIGHT_hidden = self.starting_weights(
            self.hiddenLayer, self.inputLayer)
        self.WEIGHT_output = self.starting_weights(
            self.OutputLayer, self.hiddenLayer)
        self.BIAS_hidden = np.array(
            [self.BiasHiddenValue for i in range(self.hiddenLayer)])
        self.BIAS_output = np.array(
            [self.BiasOutputValue for i in range(self.OutputLayer)])
        self.classes_number = 3

    pass

    def starting_weights(self, x, y):
        return [[2 * random.random() - 1 for i in range(x)] for j in range(y)]

    # Define activation and derivation function based on Mathematical formula
    activation = {
        'sigmoid': (lambda x: 1/(1 + np.exp(-x * 1.0))),
        'tanh': (lambda x: np.tanh(x)),
        'Relu': (lambda x: x*(x > 0)),
    }
    deriv = {
        'sigmoid': (lambda x: x*(1-x)),
        'tanh': (lambda x: 1-x**2),
        'Relu': (lambda x: 1 * (x > 0))
    }

    # Define Backpropagation process algoritm
    def Backpropagation_Algorithm(self, x):
        DELTA_output = []

        # Stage 1 - Error: OutputLayer
        ERROR_output = self.output - self.OUTPUT_L2
        DELTA_output = ((-1)*(ERROR_output) * self.deriv(self.OUTPUT_L2))

        arrayStore = []

        # Stage 2 - Update weights OutputLayer and HiddenLayer
        for i in range(self.hiddenLayer):
            for j in range(self.OutputLayer):
                self.WEIGHT_output[i][j] -= (self.learningRate *
                                             (DELTA_output[j] * self.OUTPUT_L1[i]))
                self.BIAS_output[j] -= (self.learningRate * DELTA_output[j])

        # Stage 3 - Error: HiddenLayer
        delta_hidden = np.matmul(
            self.WEIGHT_output, DELTA_output) * self.deriv(self.OUTPUT_L1)

        # Stage 4 - Update weights HiddenLayer and InputLayer(x)
        for i in range(self.OutputLayer):
            for j in range(self.hiddenLayer):
                self.WEIGHT_hidden[i][j] -= (self.learningRate *
                                             (delta_hidden[j] * x[i]))
                self.BIAS_hidden[j] -= (self.learningRate * delta_hidden[j])

    # Function for plotting error value for each epoch
    def show_err_graphic(self, v_error, v_epoch):
        plt.figure(figsize=(9, 4))
        plt.plot(v_epoch, v_error, color="b", marker=11)
        plt.xlabel("Number of Epochs")
        plt.ylabel("Squared error (MSE) ")
        plt.title("Error Minimization")
        plt.show()

    # Define predict function for prediction test data
    def predict(self, X, y):
        my_predictions = []

        # Just doing Forward Propagation
        forward = np.matmul(X, self.WEIGHT_hidden) + self.BIAS_hidden
        forward = np.matmul(forward, self.WEIGHT_output) + self.BIAS_output

        for i in forward:
            my_predictions.append(max(enumerate(i), key=lambda x: x[1])[0])

        # Print predicted value
        print(" Number of Sample  | Class |  Output  | Hoped Output")
        for i in range(len(my_predictions)):
            if (my_predictions[i] == 0):
                print("id:{}    | Iris-Setosa  |  Output: {} | Hoped Output:{}  ".format(i,
                      my_predictions[i], y[i]))
            elif (my_predictions[i] == 1):
                print("id:{}    | Iris-Versicolour    |  Output: {} | Hoped Output:{} ".format(
                    i, my_predictions[i], y[i]))
            elif (my_predictions[i] == 2):
                print("id:{}    | Iris-Iris-Virginica   |  Output: {} | Hoped Output:{} ".format(
                    i, my_predictions[i], y[i]))

        return my_predictions
        pass

    # Define fit function for training process with train data
    def fit(self, X, y):
        count_epoch = 1
        total_error = 0
        n = len(X)
        epoch_array = []
        error_array = []
        W0 = []
        W1 = []
        while (count_epoch <= self.max_epochs):
            for idx, inputs in enumerate(X):
                self.output = np.zeros(self.classes_number)

                # Stage 1 - (Forward Propagation)'
                self.OUTPUT_L1 = self.activation(
                    (np.dot(inputs, self.WEIGHT_hidden) + self.BIAS_hidden.T))
                self.OUTPUT_L2 = self.activation(
                    (np.dot(self.OUTPUT_L1, self.WEIGHT_output) + self.BIAS_output.T))

                # Stage 2 - One-Hot-Encoding
                if (y[idx] == 0):
                    self.output = np.array([1, 0, 0])  # Class1 {1,0,0}
                elif (y[idx] == 1):
                    self.output = np.array([0, 1, 0])  # Class2 {0,1,0}
                elif (y[idx] == 2):
                    self.output = np.array([0, 0, 1])  # Class3 {0,0,1}

                square_error = 0
                for i in range(self.OutputLayer):
                    erro = (self.output[i] - self.OUTPUT_L2[i])**2
                    square_error = (square_error + (0.05 * erro))
                    total_error = total_error + square_error

                # Backpropagation : Update Weights
                self.Backpropagation_Algorithm(inputs)

            total_error = (total_error / n)

            # Print error value for each epoch
            if ((count_epoch % 50 == 0) or (count_epoch == 1)):
                print("Epoch ", count_epoch, "- Total Error: ", total_error)
                error_array.append(total_error)
                epoch_array.append(count_epoch)

            W0.append(self.WEIGHT_hidden)
            W1.append(self.WEIGHT_output)

            count_epoch += 1

        self.show_err_graphic(error_array, epoch_array)

        # Print weight Hidden layer acquire during training
        print('')
        print('weight value of Hidden layer acquire during training: ')
        print(W0[0])

        # Plot weight Output layer acquire during training
        print('')
        print('weight value of Output layer acquire during training: ')
        print(W1[0])

        return self
