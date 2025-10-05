''' 
Ques1

a) The loss function: L(w) = w^2 + 2w + 5 is convex since when taken a double derivative of it
it returns 2 and not zero. Any function which has a non zero double derivative is a convex function

b) a very small learning parameter would not guarenty an optimal solution since the steps would
be too small we might never reach or might get stuck in a local minima. At the same time, it
would take a lot of iterations for the processor, putting excess load on the CPU.

Ques 2

a) False, 

b) False, if loss is increasing, the step size alpha is likely too large; you should decrease it. Since, 
otherwise we would never converge to the answer and rather just keep going over it.

c) True, with a sufficiently small constant alpha, gradient descent converges to a global minimum.

Ques 3

a) False. Zero squared loss is possible if every training point lies exactly on some line.
b) True. all residuals are zero, every training example sits perfectly on a single straight line.
c) False. Zero training error doesnt imply perfect generalization to new data.
d) False. The least-squares loss is convex (quadratic), so gradient descent doesnt get stuck in local minima.

Ques 4
a) under fit, the curve barely fits the points leading to a huge loss function value.
b) good fit
c) good fit
d) over fit, the curve touches every point. although, it has a very low loss function value,
the curve might not be able to predict new data

'''

# Ques 7

import numpy as np
import matplotlib.pyplot as plt

# Load the data (use raw string or forward slashes to avoid escape issues)
data = np.loadtxt(r"class\ex1data1.txt", delimiter=',')  # <-- keep your folder layout
U = data[:, 0]
y = data[:, 1]

# Plot the data
plt.figure()
# plt.xlim([0, 100])
# plt.ylim([-5, 100])
plt.plot(U, y, 'rx', markersize=10, linewidth=3)  # red 'x' markers
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.grid(True)

# Build design matrix with bias column of ones
X = np.column_stack((np.ones(len(U)), U))              # add column of ones to X for bias term
# Solve (X^T X) w = X^T y  without forming an explicit inverse
w = np.linalg.solve(X.T @ X, X.T @ y)     # w = [w0, w1], @ is for matrix multiplication
w0, w1 = w

y_fit  = w0 + w1 * U    # predicted values
plt.plot(U, y_fit, 'g:',linewidth=1, label='Normal Equations Fit')  # best fit line, g: means a dotted green line

val_32_y = w[0] + w[1] * 3.5
val_70_y = w[0] + w[1] * 7
print(val_32_y, val_70_y)
plt.plot(3.5, val_32_y, 'bo', markersize=6, label=f'Prediction for 35,000 population= {val_32_y*10000:.0f}')
plt.plot(7, val_70_y, 'mo', markersize=6, label=f'Prediction for 70,000 population= {val_70_y*10000:.0f}')

plt.legend()
plt.tight_layout()

# Save plot for your Word doc
plt.savefig("hw2_10_2.png", dpi=150)
plt.show()



'''
Ques 8


import numpy as np 
# Define a function to compute the loss 
def compute_loss(U, y, w): 
    errors = U @ w - y  # calculates the errors vector
    #print(errors)
    loss = (1/(2*len(y))) * np.sum(errors**2)
    return loss
# Load the data from a text file 
data = np.loadtxt('class\ex1data1.txt', delimiter=',') 
U = data[:, 0] 
y = data[:, 1] 
# Add a column of ones to U for the bias term 
Udata = np.column_stack((np.ones(len(U)), U)) 
# Initialize fitting parameters to zero 
w = np.zeros(2) 
# Call the compute_loss function 
loss = compute_loss(Udata, y, w) 
print("Loss:", loss) 

'''

"""import numpy as np 
import matplotlib.pyplot as plt
# Define a function to compute the loss 
def compute_loss(U, y, w): 
    errors = U @ w - y  # calculates the errors vector
    #print(errors)
    loss = (1/(2*len(y))) * np.sum(errors**2)
    return loss  
# Define a function to perform gradient descent 

def gradient_descent_linear(U,y,winit,alpha,iterations): 
    loss_history = np.zeros(iterations) 
    w = winit # Initialize the parameter vector 
    for i in range(iterations): 
    # Calculate new values of w 
        yhat = U @ w  # predicted values
        w[0] = w[0] - alpha * (1/len(y)) * np.sum(yhat - y)
        w[1] = w[1] - alpha * (1/len(y)) * np.sum((yhat - y) * U[:,1])      # np.sum gives a scalar
        loss_history[i] = compute_loss(U,y,w) 
    return w,loss_history 

# Load the data from a text file 
data = np.loadtxt('class\ex1data1.txt', delimiter=',') 
U = data[:, 0] 
y = data[:, 1] 
# Add a column of ones to U for the bias term 
Udata = np.column_stack((np.ones(len(U)), U)) 
# Initialize fitting parameters to zero 
winit = np.zeros(2) 
# Number of training iterations 
iterations = 1500 
# Learning rate 
alpha = 0.01 
# Call the compute_loss function 
w,loss_history = gradient_descent_linear(Udata,y,winit,alpha,iterations) 
# Print out w 
print(w) 
# Plot data and best fit lines... 
plt.plot(U, y, 'rx', markersize=10, linewidth=3)  # red 'x' markers
plt.plot(U, Udata @ w, 'g:',linewidth=1, label='Gradient Descent Fit')  # best fit line, g: means a dotted green line

# Ques 10

val_32_y = (w[0] + w[1] * 3.5)
val_70_y = (w[0] + w[1] * 7)
print(val_32_y, val_70_y)
plt.plot(3.5, val_32_y, 'bo', markersize=6, label=f'Prediction for 35,000 population= {val_32_y*10000:.0f}')
plt.plot(7, val_70_y, 'mo', markersize=6, label=f'Prediction for 70,000 population= {val_70_y*10000:.0f}')


# plt.xlim([0, 100])
# plt.ylim([-5, 100])
plt.ylabel('Profit in $10,000s')
plt.xlabel('Population of City in 10,000s')
plt.grid(True)
plt.legend()
# plt.savefig("hw_2_ex10_1.png", dpi=150)
plt.show()"""

"""# Ques 11

import numpy as np 
import matplotlib.pyplot as plt
# Normalize features to have zero mean and standard deviation of 1 
def feature_normalize(U): 
    mean = np.mean(U, axis=0)
    std = np.std(U, axis=0, ddof=0) # ddof is 0 since we want 1/N not 1/(N-1) to calculate std = root of 1/N * sum(xi - mean)^2
    Unorm = (U-mean)/std
    return Unorm   
# Define a function to compute the loss 
def compute_loss(U, y, w): 
    errors = U @ w - y  # calculates the errors vector
    #print(errors)
    loss = (1/(2*len(y))) * np.sum(errors**2)
    return loss
# Define a function to perform gradient descent 
def gradient_descent_linear(U,y,winit,alpha,iterations): 
    loss_history = np.zeros(iterations) 
    w = winit # Initialize the parameter vector 
    for i in range(iterations): 
    # Calculate new values of w 
        yhat = U @ w  # predicted values
        w[0] = w[0] - alpha * (1/len(y)) * np.sum(yhat - y)
        w[1] = w[1] - alpha * (1/len(y)) * np.sum((yhat - y) * U[:,1])      # np.sum gives a scalar
        w[2] = w[2] - alpha * (1/len(y)) * np.sum((yhat - y) * U[:,2])
        loss_history[i] = compute_loss(U,y,w) 
    return w,loss_history 
# Load the data from a text file 
data = np.loadtxt('class\ex1data2.txt', delimiter=',') 
U = data[:, :2] 
y = data[:, 2] 
# Normalize the data 
Unorm = feature_normalize(U) 
# Add a column of ones to U for the bias term 
Udata = np.column_stack((np.ones(len(Unorm)), Unorm)) 
# Initialize fitting parameters to zero 
winit = np.zeros(3) 
# Number of training iterations 
iterations = 400
# Learning rate 
alpha = 0.01 
# Call the compute_loss function 
w,loss_history = gradient_descent_linear(Udata,y,winit,alpha,iterations) 
# Print out w 
print(w) 
# Plots… 
w_ne = np.linalg.lstsq(Udata, y, rcond=None)[0]
print("GD w:", w)
print("NE w:", w_ne)
plt.plot(range(iterations), loss_history, 'b-', linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss vs. Iteration ')
plt.grid(True)
plt.savefig("hw_2_ex11.png", dpi=150)
plt.show()"""