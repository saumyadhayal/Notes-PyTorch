import pandas as pd
#--------------------------------------
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn, optim     # nn for neural network stuff, optim for optimizing the dataset
from sklearn.model_selection import train_test_split    # only used for splitting data

# settings

device = "cuda" if torch.cuda.is_available() else "cpu"     # run on GPU if available
torch.random.manual_seed(0) 
rng = np.random.default_rng(0)       # NumPy's modern RNG
'''
# creating a toy regression dataset for polynomial regression
n = 200     # number of samples
x = np.sort(rng.uniform(-3, 3, size=n))     # inputs uniformly in [-3, 3], sorted for nicer plots

# ground-truth function we will try to learn
def true_fn(t):
    return 0.5*t**3 - 2*t**2 + 1.5*t - 1    # taking random coefficients

y = true_fn(x) + rng.normal(0, 5, size=n)          # add Gaussian noise to make realistic(mean=0, std=5)

plt.scatter(x, y, s=12, alpha=0.6, label="data") # alpha = 0.6 transparency level (0 = fully transparent, 1 = fully opaque).
# 0.6 means 60% opaque, so overlapping points are easier to see.

xs = np.linspace(x.min(), x.max(), 400)            # 400 linear spacing data points between x.min and x.max
plt.plot(xs, true_fn(xs), label="true function", linewidth=2)
plt.legend(); 
plt.title("Data and ground truth"); 
plt.show()
'''

n = 200                                            # number of samples
x = np.sort(rng.uniform(-3, 3, size=n))            # inputs uniformly in [-3, 3], sorted for nicer plots

# ground-truth function we will try to learn
def true_fn(t):
    return 0.5*t**3 - 2*t**2 + 1.5*t - 1

y = true_fn(x) + rng.normal(0, 5, size=n)          # add Gaussian noise (mean=0, std=5)


# splitting the dataset into training and test sets
Xtr, Xval, ytr, yval = train_test_split(x, y, test_size=0.25, random_state=0)   # splits arrays into train (75%) and validation (25%).
Xtr_raw = Xtr.copy()
Xval_raw = Xval.copy()

# making the tensors array
def polynomial_features(array_1: np.array, degree: int, include_bias: bool = True) -> np.array:
    """
    Adding generate polynomial features for a given array up to a specified degree.
    making it in the form y = mx + b, where bias is the contant b and m is the slope.

    Parameters:
    - array_1: Input array of shape (n_samples,).
    - degree: The degree of the polynomial features to generate.
    - include_bias: If True, includes a bias term (constant term).
    
    Returns:
    - A 2D numpy array of shape (n_samples, degree + 1) containing polynomial features in each column
    and the bias term in the first column if include_bias is True, otherwise each set has 1 as first feature.

    """
    cols = [np.ones_like(array_1)] if include_bias else []  # bias column of 1s (x^0)
    ''' 
        a = [1, 1, 1]        # bias term
        b = [2, 3, 4]        # x
        c = [4, 9, 16]       # x^2... continues for higher degrees
        np.vstack([a, b, c])
          array([[ 1,  1,  1],
                 [ 2,  3,  4],
                 [ 4,  9, 16]])

        then we transpose it to get the features in columns and then collect rows as features
        np.vstack([a, b, c]).T

        array([[1., 2., 4.],
               [1., 3., 9.]])
                 
    '''
    for d in range(1, degree + 1):
        cols.append(array_1 ** d)

    x = np.vstack(cols).T  # stack columns vertically and transpose to get features in columns
    return torch.from_numpy(x).float() # convert to PyTorch tensor and return as float

degree = 12
Xtr = polynomial_features(Xtr, degree)  # training set features
Xval = polynomial_features(Xval, degree)  # validation set features
ytr = torch.from_numpy(ytr).float().unsqueeze(1).to(device)  # training set targets
yval = torch.from_numpy(yval).float().unsqueeze(1).to(device)  # validation set targets
# all values are now torch vectors and put to device (GPU or CPU).
# unsqueeze function shapes column into a column vector (n, 1) instead of (n,).

model = nn.Linear(Xtr.shape[1], 1, bias=False).to(device)
# linear function gives us the values for weights and bias (if bias = True)
# here we set bias = False because we already have a bias term in our polynomial features, which is 1
# Xtr.shape[1] = number of features (degree + 1 if bias included)
# output (second argument) is the number of output features we want from existing ones, here we want 1 output (y value)
def fit(model, Xtr, ytr, Xval, yval, *, l2=0.0, l1=0.0, lr=0.05, epochs=3000):
    with torch.no_grad():   # we don't want to track gradients for this operation
        model.weight.uniform_(-1, 1)   # initialize weights uniformly in [-1, 1]
        # if we don't put the leading _ then it won't update the tensor
        # gradient = slope of the loss function with respect to weights
        # when we use loss.backward(), it computes the gradients of the loss function with respect to all the weights
        # and then optimizer (like optim.SGD) uses these gradients to update the weights so that the loss function decreases.

    opt = optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    # we use l2 to shrink weights little by little towards zero, this is called weight decay.
    # l2 prevents the model from “memorizing” noise and forces it to learn smoother patterns.
    # Loss(new) = Loss(old) + weight_decay * (sum of all weights)     this is done to prevent overfitting

    mse = nn.MSELoss() # Mean Squared Error loss function, which is the average of the squared differences between predicted and actual values.
    hist = {"train": [], "val": []} # keeping track of training and validation loss

    for _ in range(epochs):
        model.train()
        pred = model(Xtr) # forward pass: compute predicted y by passing Xtr through the model
        loss = mse(pred, ytr) # calculate the loss between predicted and actual values

        if l1 > 0:  # if we have lasso regularization (l1 > 0), we add L1 regularization term
            l1_pen = 0.0    # L1 penalty term, which is the sum of absolute values of weights
            for p in model.parameters():           # here it's just the weight matrix
                l1_pen = l1_pen + p.abs().sum()    # sum of absolute values = L1 norm
            loss = loss + l1 * l1_pen              # total loss = MSE + λ * ||w||_1

        opt.zero_grad() # we need to clear all gradients before backward pass
        # in pytorch, each call to .backward() adds to the previous gradients.
        loss.backward() # gets gradients of the loss function w.r.t. each model parameter (weight)
        # this is where the gradients are calculated, and they are stored in the .grad attribute of each parameter.
        opt.step() # Updates the model’s parameters using the gradients stored in .grad

        model.eval()
        with torch.no_grad():
            hist["train"].append(mse(model(Xtr), ytr).item())
            # mse = nn.MSELoss() calculates the mean squared error between: model(Xtr) which is the predicted values and ytr which is the actual values
            # mse(model(Xtr), ytr) gives PyTorch tensor containing the average squared error over all training samples.
            # .item() converts the tensor to a Python number (float)
            hist["val"].append(mse(model(Xval), yval).item())
    return hist

model0 = nn.Linear(Xtr.shape[1], 1, bias=False).to(device)   # fresh model
hist0  = fit(model0, Xtr, ytr, Xval, yval, l2=0.0, l1=0.0, epochs=4000, lr=0.05)

def predict_curve(model, degree, xs):
    """Helper: build polynomial features for xs and return model predictions as a 1D NumPy array."""
    Xt = polynomial_features(xs, degree).to(device)
    with torch.no_grad():
        return model(Xt).cpu().numpy().ravel()

xs_plot = np.linspace(x.min(), x.max(), 400)
y_no = predict_curve(model0, degree, xs_plot)

plt.figure(figsize=(8, 5))
print("Xtr shape:", Xtr.shape)
print("ytr shape:", ytr.shape)
# plt.scatter(Xtr, ytr, s=12, alpha=0.6, label="train")
# plt.scatter(Xval, yval, s=12, alpha=0.6, label="val")

plt.scatter(Xtr_raw, ytr.cpu().numpy().ravel(), s=12, alpha=0.6, label="train")
# using raw Xtr_raw and Xval_raw to show original x values before polynomial features transformation
# using ytr.cpu().numpy().ravel() to convert the tensor to a NumPy array and flatten it to 1D for plotting
plt.scatter(Xval_raw, yval.cpu().numpy().ravel(), s=12, alpha=0.6, label="val")
plt.plot(xs_plot, true_fn(xs_plot), linewidth=2, label="true")
plt.plot(xs_plot, y_no, linewidth=2, label="no reg")
plt.title(f"No regularization (degree={degree})")
plt.legend(); plt.tight_layout(); plt.show()

print("MSE (no-reg) | train:", hist0["train"][-1], "| val:", hist0["val"][-1])


#---------------------------------------------------------------------------------------
# coefs = [17, 3, -1]   # Coefficients for the polynomial: 15 + 1*x - 1*x^2
# # here it starts with a constant term, then x , x^2, x^3, etc.

# def polynomial(coefs, x):
#     n = len(coefs)  # number of coefficients
#     return sum([coefs[i]*x**i for i in range(n)]) # returns the value of the polynomial at x

# def draw_polynomial(coefs):
#     n = len(coefs)
#     x = np.linspace(-5, 5, 1000)
#     plt.ylim(-20,20)
#     plt.plot(x, sum([coefs[i]*x**i for i in range(n)]), linestyle='dotted', color='blue' )
#     plt.show()

# draw_polynomial(coefs)
#----------------------------------------------------------------------------------------


modelR = nn.Linear(Xtr_t.shape[1], 1, bias=False).to(device)
histR  = fit(modelR, Xtr_t, ytr_t, Xval_t, yval_t, l2=1e-3, l1=0.0, epochs=4000, lr=0.05)
y_rid  = predict_curve(modelR, degree, xs_plot)

plt.figure(figsize=(8, 5))
plt.plot(xs_plot, true_fn(xs_plot), linewidth=2, label="true")
plt.plot(xs_plot, y_no,  linewidth=2, label="no reg")
plt.plot(xs_plot, y_rid, linewidth=2, label="ridge (L2)")
plt.title("Effect of L2 regularization")
plt.legend(); plt.tight_layout(); plt.show()

print("MSE (ridge)  | train:", histR["train"][-1], "| val:", histR["val"][-1])
#---------------------------------------------------------------------------------------

modelL = nn.Linear(Xtr_t.shape[1], 1, bias=False).to(device)
histL  = fit(modelL, Xtr_t, ytr_t, Xval_t, yval_t, l2=0.0, l1=1e-4, epochs=4000, lr=0.05)
y_las  = predict_curve(modelL, degree, xs_plot)

plt.figure(figsize=(8, 5))
plt.plot(xs_plot, true_fn(xs_plot), linewidth=2, label="true")
plt.plot(xs_plot, y_no,  linewidth=2, label="no reg")
plt.plot(xs_plot, y_las, linewidth=2, label="lasso (L1)")
plt.title("Effect of L1 regularization")
plt.legend(); plt.tight_layout(); plt.show()

print("MSE (lasso)  | train:", histL["train"][-1], "| val:", histL["val"][-1])
#---------------------------------------------------------------------------------------

def wvec(m): 
    return m.weight.detach().cpu().numpy().ravel()   # extract weights as 1D NumPy array

print("Weights (no reg):", wvec(model0))             # [w0 (bias), w1 (x), w2 (x^2), ...]
print("Weights (ridge) :", wvec(modelR))             # smaller magnitudes overall
print("Weights (lasso) :", wvec(modelL))             # some may be exactly 0


#---------------------------------------------------------------------------------------

def eval_combo(deg, l2=0.0, l1=0.0, epochs=2500):
    Xtr_t = poly_features(Xtr, deg).to(device)
    Xval_t = poly_features(Xval, deg).to(device)
    ytr_t  = torch.from_numpy(ytr).float().unsqueeze(1).to(device)
    yval_t = torch.from_numpy(yval).float().unsqueeze(1).to(device)
    m = nn.Linear(Xtr_t.shape[1], 1, bias=False).to(device)
    h = fit(m, Xtr_t, ytr_t, Xval_t, yval_t, l2=l2, l1=l1, epochs=epochs, lr=0.05)
    return h["train"][-1], h["val"][-1]

for d in [1, 3, 5, 12]:
    tr, va = eval_combo(d, l2=0.0, l1=0.0)
    print(f"degree {d:>2} no-reg  -> train {tr:8.3f} | val {va:8.3f}")

for lam in [0.0, 1e-4, 1e-3, 1e-2]:
    tr, va = eval_combo(12, l2=lam, l1=0.0)
    print(f"degree 12 L2={lam:<6} -> train {tr:8.3f} | val {va:8.3f}")
