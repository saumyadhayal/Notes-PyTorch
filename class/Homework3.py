"""# Q5: 5th-order polynomial via normal equations (with feature scaling)
import numpy as np
import matplotlib.pyplot as plt

# Data
u_raw = np.array([4, 62, 120, 180, 242, 297, 365])
y     = np.array([2720, 1950, 1000, 1150, 1140, 750, 250])

# --- scale u to [-1, 1] to fix conditioning ---
umin, umax = u_raw.min(), u_raw.max()
u = (2*(u_raw - umin)/(umax - umin)) - 1.0   # now in [-1,1]

# matrix [1, u, ..., u^5]
U = np.column_stack([u**k for k in range(6)])

# Normal equations (λ = 0)
A = U.T @ U
b = U.T @ y
w = np.linalg.solve(A, b)

raw_data = np.linspace(0, 400, 400)
less_u = (2*(raw_data - umin)/(umax - umin)) - 1.0
UU = np.column_stack([less_u**k for k in range(6)])
yhat = UU @ w

mse = np.mean((y - U @ w)**2)
print("MSE =", mse)

plt.plot(u_raw, y, 'o', color='red', markersize=6)
plt.plot(raw_data, yhat, linewidth=2)
plt.grid(True)
plt.title(f"5th Order Fit, $\\lambda$ = 0, MSE={mse:.4f}")
plt.xlabel("Day of Year"); plt.ylabel("Bank Account Balance ($)")
plt.savefig("hw3_Q5.png", dpi=300)
plt.show()
"""
"""
import numpy as np
import matplotlib.pyplot as plt

u = np.array([0, 0.2072, 0.3494, 0.4965, 0.6485, 0.7833, 0.9400])
y = np.array([2.150, 1.541, 0.790, 0.909, 0.901, 0.593, 0.198])

# ----- [1, u, u^2, ..., u^5] -----
U  = np.column_stack([u**k for k in range(6)])
data_x = np.linspace(u.min(), u.max(), 400)
data_y = np.column_stack([data_x**k for k in range(6)])

# lambda = 0 (unregularized)
# w = (U^T U)^(-1) U^T y
w_unreg = np.linalg.solve(U.T @ U, U.T @ y) # solves for w 
yhat_unreg = data_y @ w_unreg

lam = 1e-3
I = np.eye(U.shape[1])      # identity matrix
I[0, 0] = 0.0                # exclude bias from penalty
A = U.T @ U + lam * I   # regularized normal eqn matrix otherwise same as unreg
b = U.T @ y
w_ridge = np.linalg.solve(A, b)
yhat_ridge = data_y @ w_ridge

# mse for both models
mse_unreg = np.mean((y - U @ w_unreg)**2)
mse_ridge = np.mean((y - U @ w_ridge)**2)

plt.figure()
plt.plot(u, y, 'o', markersize=8, label='data')
plt.plot(data_x, yhat_unreg, linewidth=2, label='5th-order ($\\lambda$=0)')
plt.plot(data_x, yhat_ridge, '--', linewidth=2, label='5th-order ($\\lambda$=0.001)')
plt.grid(True)
plt.title('MSE of the regularized model')
plt.xlabel('Percent of Semester')
plt.ylabel('Bank Balance ($K)')
plt.legend()
plt.savefig("hw3_Q6.png", dpi=300)
plt.show()"""
"""
import numpy as np
import matplotlib.pyplot as plt
# percent of way in semester
u =np.array([0,0.2072,0.3494,0.4965,0.6485,0.7833,0.9400])
# bank balance ($K)
y =np.array([2.150,1.541,0.790,0.909,0.901,0.593,0.198])

uu = np.linspace(u.min(), u.max(), 400)  # x values for the curve

Ns   = [1, 3, 5]
lams = [0.0, 1e-3, 1.0]

fig, axes = plt.subplots(3, 3)

for i in range(len(Ns)):
    N = Ns[i]
    data_x  = np.column_stack([u**k for k in range(N+1)])
    data_y = np.column_stack([uu**k for k in range(N+1)])
    for j in range(len(lams)):
        lam = lams[j]
        ax = axes[i, j]
        # Regularized normal equation: (U^T U + λI*)w = U^T y
        I = np.eye(data_x.shape[1])
        I[0, 0] = 0.0           # exclude bias from penalty
        A = data_x.T @ data_x + lam * I
        b = data_x.T @ y
        w = np.linalg.solve(A, b)
        yhat = data_y @ w

        ax.plot(u, y, 'o', label='data')
        ax.plot(uu, yhat, 'r', lw=2)
        ax.set_title(f'N={N}, $\\lambda$={lam}')
        ax.grid(True)

        ax.set_xlabel('u')
        ax.set_ylabel('y')

plt.tight_layout()
plt.savefig("hw3_Q7.png", dpi=300)
plt.show()
"""

import numpy as np 
import matplotlib.pyplot as plt 
import sklearn.linear_model

np.random.seed(2000)  # Random number generator seed 
mu = np.array([0, 0]) 
sigma = np.array([[4, 1.5], [1.5, 2]]) 
r = np.random.multivariate_normal(mu, sigma, 50)  # Create two features, 50 samples of each 
y = r[:, 0,None] 
u = (np.pi * (np.arange(50) + 1) / 20).reshape(-1, 1)  # Scale u for sin 
np.random.shuffle(u) 
y = 10 * np.sin(u) * (4 + y)  # Add some curvature 
y = y + u*4  # Gradually rise over time 
utrain = u[::2]  # Odd samples for train 
ytrain = y[::2] 
utest = u[1::2]  # Even samples for test 
ytest = y[1::2] 
Utrain = np.column_stack([utrain**1, utrain**2, utrain**3])
# plt.figure(1) 
# plt.clf() 
# plt.plot(utrain, ytrain, 'rs', markersize=10, linewidth=3, markerfacecolor='c', markeredgecolor='b') 
# plt.plot(utest, ytest, 'ro', markersize=10, linewidth=3, markerfacecolor='m', markeredgecolor='r') 
# plt.grid(True) 
# plt.legend(['train', 'test']) 
# plt.savefig('hwk3_problem8_data.png') 

model = sklearn.linear_model.LassoCV(cv=2).fit(Utrain,ytrain.ravel()) 
# plt.figure() 

# plt.semilogx(model.alphas_,np.mean(model.mse_path_,axis=1)) 
# plt.gca().invert_xaxis()
# plt.xlabel('$\lambda$') 
# plt.ylabel('Validation MSE') 
# plt.savefig('hwk3_problem8_lasso_path.png')
# plt.grid(True)
# plt.show()

uu = np.linspace(u.min(), u.max(), 400).reshape(-1, 1)  # 400 evenly spaced points
UU = np.column_stack([uu, uu**2, uu**3])

y_curve = model.predict(UU)

plt.figure()
plt.plot(utrain, ytrain, 's', ms=10, mfc='c', mec='b', ls='none', label='train')
plt.plot(utest,  ytest,  'o', ms=10, mfc='m', mec='r', ls='none', label='test')
plt.plot(uu, y_curve, 'k-', lw=2, label='LassoCV fit')  # solid black curve
plt.grid(True)
plt.xlim(0, 8); plt.ylim(-50, 100)
plt.xlabel('u'); plt.ylabel('y')
plt.legend()
plt.title('Problem 8: Model Overlay')
plt.savefig('hwk3_problem8_last.png', dpi=300)
plt.show()

Utest  = np.column_stack([utest**1,  utest**2,  utest**3])
def mse(y_true, y_pred):
    return np.mean((y_true.ravel() - y_pred.ravel())**2)

chosen_lambda = model.alpha_
train_mse = mse(ytrain, model.predict(Utrain))
test_mse  = mse(ytest,  model.predict(Utest))

print(f"Chosen $\lambda$ (alpha_): {chosen_lambda:.6g}")
print(f"Train MSE: {train_mse:.4f}")
print(f"Test  MSE: {test_mse:.4f}")