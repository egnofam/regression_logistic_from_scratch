import numpy as np
import matplotlib.pyplot as plt


#
def generate_dataset() -> None:
    """
        function to generate datasets
        :return:  features and targets
    """
    nb_rows = 100  # generate number of rows
    covid_case = np.random.randn(nb_rows, 2) + [-3, -3]  # generate covid cases
    non_covid_case = np.random.randn(nb_rows, 2) + [3, 3]  # generate non covid cases
    target_covidcase = np.zeros((covid_case.shape[0])).reshape(nb_rows, 1)  # generate target for covid cases
    target_non_covidcase = np.ones((non_covid_case.shape[0])).reshape(nb_rows, 1)  # generate target for non covid cases
    target = np.concatenate((target_covidcase, target_non_covidcase), axis=0)  # concatenate target for covid cases and non covid cases
    patients = np.concatenate((covid_case, non_covid_case), axis=0)  # concatenate patients datasets
    return patients, target


def plot_figure(features, y):
    """
        function to plot figure
        :param data: should have two features
        :param y: target value
        :return: none
    """
    plt.scatter(features[:, 0], features[:, 1], c=y)
    plt.show()


def init_wieights_bias():
    """
        init weights and bias
        :return: none
    """
    weights = np.random.randn(2)
    bias = 0
    return weights, bias


def pre_activation(features, weights, bias):
    """
        compute the preactivation
        :param features: features of the data
        :param targets: target of the data
        :param weights: weights of features
        :param bias: bias
        :return: the preactivation
    """
    z = np.dot(features, weights) + bias
    return z


def sigmoid(z):
    s = (1/(1 + np.exp(-z)))
    return s


def activation(z):
    """
        function to compute the activation
        :param z: the preactivation
        :return: the activation
    """
    return sigmoid(z)


def predict(z):
    """
        function to compute the activation
        :param z: the preactivation
        :return: the prediction
    """
    return np.round(sigmoid(z))


# initialise x and y
x = np.random.randint(-10, 10)
y = np.random.randint(-10, 10)


def function_cout(predictions, targets):
    return np.mean((predictions - targets)**2)


def deriv(z):
    return sigmoid(z) * (1 - sigmoid(z))


def train(features, targets, weights, bias, lr, epochs):
    z = pre_activation(features, weights, bias)
    pred = predict(z)
    # print("z initial = ", z)
    print("accuracy = ", np.mean(pred == targets))
    for epoch in range(epochs):
        if epoch % 10 == 0:
            predictions = predict(z)
            print("cost = ", function_cout(predictions, targets))
        weights_gradients = np.zeros(weights.shape)
        bias_gradients = 0
        # go trough each row in the dataset
        for feature, target in zip(features, targets):
            z = pre_activation(feature, weights, bias)
            prediction = predict(z)
            # print("prediction = ", prediction)
            # print("feature",feature)
            # update gradients
            weights_gradients += (prediction - target) * deriv(z) * feature
            bias_gradients += (prediction - target) * deriv(z)
            # print("weights init",(prediction - target) * deriv(z) * feature, "weights gradients ", weights_gradients)

            # print("bias gradients ", bias_gradients)
            # print("preactivation = ", z, "prediction = ", prediction, " target = ", target, "weights init",
            #       (prediction - target) * deriv(z) * feature, " weights = ", weights_gradients, "bias gradients ", bias_gradients)

            # return

        # update variables
        weights = weights - lr * weights_gradients
        bias = bias - lr * bias_gradients

        z = pre_activation(features, weights, bias)
        pred = predict(z)
        # print("z initial = ", z)
        print("accuracy f = ", np.mean(pred == targets))


if __name__ == "__main__":

    # get features, targets , weights, bias
    features, targets = generate_dataset()

    # print("shape ", features.shape, targets.shape)
    weights, bias = init_wieights_bias()
    zz = pre_activation(features, weights, bias)
    aa = activation(zz)
    # print("weights : ", weights)

    # predictions is the same as activation
    predictions = aa

    # print accuracy
    accuracy = np.mean(predictions == targets)
    # print("accuracy = ", accuracy)

    # train the model

    train(features, targets, weights, bias, lr=0.1, epochs=4)

    # plot figure

    #plot_figure(features, targets)



