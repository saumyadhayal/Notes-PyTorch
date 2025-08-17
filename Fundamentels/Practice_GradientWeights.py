import torch
import torch.nn as nn
import torch.optim as optim

# One weight model: y_hat = w*x
w = torch.tensor([1.0], requires_grad=True)   # start at 1.0
x = torch.tensor([2.0])
y_true = torch.tensor([10.0])

# prediction and loss
y_hat = w * x   # y_hat = 1.0 * 2.0 = 2.0
loss = (y_hat - y_true)**2   # MSE = (2.0 - 10.0)**2 = 64.0

loss.backward()              # compute gradient = dLoss/dw
print("Gradient:", w.grad)   # shows dLoss/dw = 2*x*(w*x - y_true) = 2*2*(2 - 10) = -32.0

opt = optim.SGD([w], lr=0.1)
opt.step()                   # use gradient to update weight = w - lr * gradient = 1.0 - 0.1 * (-32.0) = 1.0 + 3.2 = 4.2
print("Updated weight:", w)

''' 
when running this code in a loop, the weight keeps updating. the number of epoch decides how 
many times we update it and how close we get to the true value.
'''

# One-weight model: y_hat = w * x
w2 = torch.tensor([0.5], requires_grad=True) # random start value for weight
x2 = torch.tensor([4.0])    # x value
y_true2 = torch.tensor([20.0])                # true output = 20 for x2

opt2 = optim.SGD([w2], lr=0.05)               # small learning rate
mse2 = nn.MSELoss()

history2 = {"step": [], "w": [], "grad": [], "loss": []}

num_steps2 = 20 # number of steps to run like epochs
for step in range(1, num_steps2 + 1):
    # ---- forward
    y_hat2 = w2 * x2
    loss2 = mse2(y_hat2, y_true2)   # MSE loss = (y_hat2 - y_true2)^2

    # ---- backward
    opt2.zero_grad()    # reset gradient to zero before backward pass
    loss2.backward()    # compute gradient dLoss/dw = 2*x*(w*x - y_true)
    # it calculated gradient with w2, since we set requires_grad=True for w2 and treated x2 as a constant
    '''
    when we call backwards function:
    1. starts at loss2
    2. works backwards, first (y_hat2 - y_true2)^2 and then y_hat2 = w2 * x2
    3. uses chain rule:
        loss = (w2 * x2 - y_true2)^2
         dLoss/dw2 = 2 * (w2 * x2 - y_true2) * (x2 * 1 - 0) --- derivative of bracket since w2 becomes 1
         and its constant x2 remains, and also y_true2 is contant so it becomes 0.
    '''
    
    # ---- record stats
    history2["step"].append(step)
    history2["w"].append(w2.item())
    history2["grad"].append(w2.grad.item())
    history2["loss"].append(loss2.item())

    # ---- update
    opt2.step() # update w2 using the gradient by w2 = w2 - lr * dLoss/dw2

# Show first few (5) steps
for i in range(5):
    print("Step:", history2['step'][i])
    print("Weight:", history2['w'][i])
    print("Gradient:", history2['grad'][i])
    print("Loss:", history2['loss'][i])
    print("-" * 20)

# Print final weight
print("Final weight:", w2.item())
'''
 our final weight should be close to 5 since y_true2 = 20 and x2 = 4, so w2 should be 20/4 = 5.
 and we got value as 4.999
'''