import matplotlib.pyplot as plt
import random as rand
import numpy as np

x = np.array([1,2,3,5,6,7], dtype=float)
y = np.array([155, 197, 244, 356,407,448], dtype=float)
print("Features: ", x)
print("Labels: ", y)
# SQUARE METHOD
# learning rate is the small step size between each iteration
# features are the input data points
# labels are the target values we want to predict
# epochs are the number of times we want to iterate over the dataset
# p' = mx + p / y = bx + y
rand.seed(0)
def linear_reg(features, labels, epochs, learning_rate = 0.01):
    m = rand.random()  # taking random slope and y-intercept
    b = rand.random()  
    plt.ion()  # interactive mode: allows dynamic updates to plots
    fig, ax = plt.subplots()
    ax.scatter(features, labels, label="Data points")  # plot the training data
    ax.set_xlim(features.min(), features.max())
    ax.set_ylim(min(0, labels.min() - 30), labels.max() + 20)
    ax.set_xlabel("features")
    ax.set_ylabel("labels")
    ax.set_title("Linear Regression: Square Method")
    line, = ax.plot([], [], label="Model line")  # an empty line we'll update every step
    ax.legend(loc="upper left")

    x_line = np.array([features.min(), features.max()])
    for epoch in range(epochs):
        # Pick one random point
        i = rand.randint(0, len(features) - 1)
        x_i, y_i = features[i], labels[i]

        m, b = square(learning_rate, x_i, y_i, m, b) # update slope and y-intercept

        # 3) Redraw the line to show progress (animation)
        y_line = b + m * x_line # calculate the y values for the x_line
        line.set_data(x_line, y_line)
        plt.pause(0.01)  # small pause to visually animate
        # pause is animation speed; 0.0 is fastest (no delay)

    print("Final slope:", m)
    print("Final y-intercept:", b)
    plt.ioff()
    plt.show()
    
    return m, b
# x = feature value for one data point (x value)
# y = actual label (y value) for that data point
# b = y-intercept
# m = slope
def square(lr, x, y, m, b):
    y_hat = b + m * x  # predicted value of the intercept: y_hat = mx + b
    m = m + lr * x * (y - y_hat)  # new slope
    b = b + lr * (y - y_hat)  # new y-intercept
    return m, b

# calling the function
m_learned, b_learned = linear_reg( x, y,
    learning_rate=0.02,   # step size (try 0.001 or 0.02 to feel the difference)
    epochs=1000           # number of updates (more = smoother convergence)
)

'''
Coursera Notes for Linear Regression: Terminologies

Training set consists of data points in the form of (x, y) pairs. 
x are the features,
y are the targets or original values of function
y-hat is the predicted value of y, that is calculated using the model.
f(x) = wx + b = y-hat

Lost Function: tells the score of how well the model is performing. The lower the better
Formula: L = (1/N)(y - y-hat)^2 where N is the number of data points

This beasically is the mean squared error (MSE) function, which is the square of distance
between the predicted value and the actual value

Gradient Descent: is the algorithm used to minimize the loss function
Algorithm:
w = w - learning_rate * dL/dw       L= loass from previous step
b = b - learning_rate * dL/db
The learning rate is small due to which it pushes the values of w and b slowly towards the 
original values. If we keep the lkearning rate high then we might miss our original value and keep 
overshooting it. We dont want to make the lr too small either because then it will take too long to converge.
If the original value is to the left of the currebt value, then the derivative/slope will be negative 
and hence the value of w will increase and vice versa.

When the point gets closer, the slope becomes smaller and therefore update steps become smaller.
Batch Gradient descent: each set of the gradient descent uses all of the training examples
f = w1*x1 + w2*x2 + ... + b     ( w.x = dot product )
w = [w1, w2, w3] = parameters of the model
b is a number and x is a vector

for subtraction and additon, we can directly use the - and + for arrays with same size

Normal Equation for linear regression (Other way of gradient descendt)
'''