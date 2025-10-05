import numpy as np
import matplotlib.pyplot as mtplt
'''
# Ques 1

x1 = np.array([1,2,3])
x2 = np.array([[1],[2],[3]])
x3 = np.array([1,2,3]).T
x4 = np.array([1,2,3]).T[:,None]
print(x1,"\n", x2,"\n", x3,"\n", x4)
print("----------------------")

#Ans: 2, 4

# Ques 2

final2 = np.empty((0,3))    # creates an empty array with 0 rows and 3 columns
for i in range (1,5):
    a = np.arange(i,i+3, 1)
    # print(a)
    final2 = np.vstack([final2, a])     # stacks a on final2 and they keep adding down in the rows

print(final2)

print("----------------------")

# Ques 3

print("A. ", final2[:,2:3])
print("B. ", final2[:,2])
print("C. ", final2[:,1:3])
print("D. ", final2[:,:3])

# Ans: option C
print("----------------------")

# Ques 4

A = np.array([[1,2],[3,4]])
B = np.array([[2,2], [3,3], [4,4]])
C = np.eye(3)   # square matrix with 1 in diagonals and 0 everywhere else
D = np.array([1,2,3])
E = np.zeros((3,3))
# a) ERROR print(A*(B.T)) element type operation not possible since theyre (2,2) and (2,3). they should be same
"""
 b)
print(A@(B.T)) 
 Ans: (2,2)(2,3) gives a matrix multiplication of size (2,3):
[[ 6  9 12]
 [14 21 28]]
"""

# c) print(np.dot(A,B.T)) same as b. does a matrix multiplication
# d) ERROR print(A-B) not same size
# e) print(C@E) gives a 3x3 zero matrix
# f) print (C*E) gives a 3x3 zero matrix too
# g)
"""
print(D*(B.T))
Ans:
[[ 2  6 12]
 [ 2  6 12]]
"""
# h) print(D@E) gives a (1,3) zero matrix

d = np.array([1,2])
f = np. zeros((2,1))
#a) print(B - np.matlib.repmat(f,1,3)) invalid
# b) 
"""
b = np.concatenate((d.T, d.T*2))
Ans: [1 2 3 2 4 6] it formed a transpose first which resulted in [1,2,3] for D and multiplied by 2 for next term
and got [2,4,6] then those two were concatenated
"""
# c) 
# b = np.concatenate((d.T, d.T*2),axis=1)
# ERROR: there's only axis 0, same as column stacking but with the axis given. so each column is an axis starting with 0

# d) 
b = np.concatenate((d.T[None,:], d.T[None,:]*2),axis=1) # the [None,:] adds a new dimension of size 1
print(b)
# so basically [None,:] adds a dimention to the front. like makes (4,) to a (1,4) making [1,2,3] row to [[1,2,3]] column vector

# e) 
print(b + np.concatenate((f,f,f)))
"""
here b is:  [[1 2 3 2 4 6]]
Ans for e:
[[1. 2. 3. 2. 4. 6.]
 [1. 2. 3. 2. 4. 6.]
 [1. 2. 3. 2. 4. 6.]
 [1. 2. 3. 2. 4. 6.]
 [1. 2. 3. 2. 4. 6.]
 [1. 2. 3. 2. 4. 6.]]

 so basically f,f,f makes an array with the rows and then 
"""
print("----------------------")

# Ques 6

print(np.ones((4,1))*5)
print(np.ones((4))*5)
print(5*np.ones((1,4)).T)
# print(np.ones((4,1))@5) ERROR
print(np.eye(4)*5)

# Ans: 1,3

print("----------------------")

# Ques 7
J = np.empty((0,5))    # creates an empty array with 0 rows and 3 columns
for i in range (1,6):
    a = np.arange(i,i+5, 1)
    # print(a)
    J = np.vstack([J, a])     # stacks a on final2 and they keep adding down in the rows
# print(J)
print(J[1:4,0:5:2])
print(J[0:5:2, 1:4])
print(J[1:4, [0,2,4]])
print(np.concatenate((J[0,1:4], J[2,1:4], J[4,1:4]), axis = None))
print(np.concatenate((J[1:4,0], J[1:4,2], J[1:4,4]), axis = None))

# ans: 1,3,5
'''
'''
# Ques 8
x_vector = np.arange(0,(3/2)*np.pi,0.05)        # when we want a range of points linearly with a step size = arange
 # if we want points distributed in a space linearly then we use linspace
y_vector = np.sin(x_vector)
y2_vector = np.sinc(x_vector/np.pi)
y3_vector = np.sin(x_vector**2)

fig, ax = mtplt.subplots(figsize=(6,4))
ax.set_title("Dhayal CMPE 677, Hwk 1, Problem 7, λ=0", fontsize=12)
ax.set_xlim(0,6)
ax.set_ylim(-5, 5)
ax.set_xlabel("Time", fontsize=12)
ax.set_ylabel("Response", fontsize=12)

ax.plot(x_vector, y_vector, label="sin(x)", color="red", linestyle="-", linewidth=3)
ax.plot(x_vector, y2_vector, label="sinc(x/pi)", color="green", linestyle="-", linewidth=3)
ax.plot(x_vector, y3_vector, label="sin(x^2)", color="blue", linestyle="-", linewidth=3)

ax.grid(True)
ax.legend()
mtplt.savefig('Dhayal_FirstPlot.png', format='png')
mtplt.show()


# # Ques 9
A = np.array([[1, 0, -4, 8, 3], [4, -2, 3, 3, 1]]) 
b = np.zeros(5) 
print()
for index in range(A.shape[1]): 
    if A[0, index] > A[1, index]: 
        b[index] = A[0, index] 
    else: 
        b[index] = A[1, index] 
print(b)

# Ans: [4 0 3 8 3]



# Ques 10
# b = 3x2, a is 2x1, 
# print(np.array(np.concatenate((np.eye(2),np.zeros((2,1))),axis=1))*0 )
# print(np.eye(2,3))
# print(np.array([[0,0],[0,0], [0,0]]).T)

# Ans: c, d, e, f
'''
# """
# Ques 11
'''
Gaussian distribution is the bell like distribution of the data:
mean is where the data is centred and variance means how widely is it spreaded accross the plane

**Think of the Gaussian (normal distribution) as a "bell-shaped curve" that tells us:
    where the data is centered (the mean),
    how spread out it is (the variance).


Extending to 2 variables (multivariate Gaussian): makes a hill now since we have more parameters
so mean = [0,3] mean hill is centred at x1 = 0 and x2 = 3
and covarience matrix: tells us the shape of the hill
so [[u11, u12][u21, u22]] = (u11, u12) tell us varience of x1 and x2 (relation on diagonals)
whereas (u21, u22) = relation between the two (Off-diagonals)

PDF = Probability Density Function

mvn is like a machine representing your 2D Gaussian.
.pdf(point) → gives you the height (density) at that point
    *tells you how “dense” probability is around that point.
    *not a probability itself (since continuous variables dont have probability at one exact point).
    *but if you integrate the PDF over a region, you get probability.

mvn.pdf([x,y]) → evaluates the PDF formula at that coordinate. This is the “height” 
of the Gaussian surface at (x,y).
    Z in your code = a 2D table of those PDF values, one for each grid point.
    When you plotted contours, you visualized “all points where PDF = some constant” 
    (like elevation lines on a map)

when PDE is integrated: probability that the random variable falls inside that interval.


READ MORE ABOUT PROBABILITY DISTRIBUTION
'''

# import matplotlib.pyplot as plt 
# from scipy.stats import multivariate_normal 
# # Define mean (mu) and covariance matrix (sigma) 
# mu = np.array([0, 3]) 
# sigma = np.array([[5, -2], [-2, 2]]) 
# # Create a grid of x1 and x2 values 
# x1 = np.arange(-10, 10.1, 0.1) 
# x2 = np.arange(-10, 10.1, 0.1) 
# X1, X2 = np.meshgrid(x1, x2) 
# # Create a multivariate normal distribution 
# # algorithm that prepares the model with the mean and varience of data
# mvn = multivariate_normal(mean=mu, cov=sigma)

# # Calculate the PDF values for each point in the grid 
# # pdf here is the height when plotting the gaussian curve where we have x1 and x2 on x and y axis 
# # and the mean and varience give us the distribution of data
# # ravel function flattens the 2d or 3d matrix into 1d 
# # opposite of ravel() function is reshape(2,3) this reshapes the 1d matrix into rows,columns
# pdf_values = mvn.pdf(np.column_stack((X1.ravel(), X2.ravel()))) 
# # print(pdf_values)

# # Reshape the PDF values to match the grid shape 
# F = pdf_values.reshape(X1.shape) 
# # print(F)
# # Create a contour plot 
# # plt.contour(x1, x2, F) 
# # # Set plot attributes 
# # plt.grid(True) 
# # plt.axis('square') 
# # plt.title('Dhayal CMPE 677, Hwk 1, Problem 10', fontsize=12) 
# # # Save the plot as a PNG file 
# # plt.savefig('cmpe677_hwk1_10.png', format='png') 
# # # Show the plot 
# # plt.figure()
# '''
# marginal distribution is what you get when you only care about one variable (say X) and “ignore”
#  the other one (Y). So it's basically plotting the height vs the x and y values
# '''
# marg_y = np.trapezoid(F, x1, axis=1)   # integrates over x1 values, axis = 0 for going across column for values
# marg_x = np.trapezoid(F, x2, axis=0 )   # axis = 1 for integrate across each row, leaving you with one result per row
# plt.plot(x1, marg_x, label="f_X(x)")
# plt.plot(x2, marg_y, label="f_Y(y)")
# plt.legend()
# plt.title("Marginals for the Gaussian Distribution")
# plt.savefig('cmpe677_hwk1_10_2.png', format='png') 
# # plt.show() 

# # '''
# # b) to get the mean of the graphs, we will take the integral of x * PDF
# # since PDF is between 0 and 1
# # '''

# # mu_x = np.trapezoid(marg_x * x1, x1) # no axis needed, both are 1D

# # print(f"{x1 = }, {x1 * marg_x = }, {mu_x = }")

# # mu_y = np.trapezoid(marg_y * x2, x2)

# # print(f"{x2 = }, {x2 * marg_y = }, {mu_y = }")

# # plt.axvline(mu_x,linestyle="--", label="mu(X)", color="red")
# # plt.axvline(mu_y, linestyle="--", label="mu(y)", color="green")
# # plt.legend()
# # plt.title("Means of Marginal Distribution")
# # plt.savefig('cmpe677_hwk1_10_3.png', format='png') 
# # plt.show()

# """
'''
probability density changes along the x-axis, for different fixed values of y
for y = 0 is highest since the curve is tallest (highest probability)
but for the other y values the curves get shorter because you're slicing farther away from the peak of the hill.
'''
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal 
# Define mean (mu) and covariance matrix (sigma) 
mu = np.array([0, 0]) 
sigma = np.array([[5, -2], [-2, 2]]) 
# Create a grid of x1 and x2 values 
x = np.arange(-10, 10.1, 0.1) 
y = np.arange(-10, 10.1, 0.1) 
X, Y = np.meshgrid(x, y) 
# Create a multivariate normal distribution 
mvn = multivariate_normal(mean=mu, cov=sigma) 
# Calculate the PDF values for each point in the grid 
pdf_values = mvn.pdf(np.column_stack((X.ravel(), Y.ravel()))) 
# Reshape the PDF values to match the grid shape 
F = pdf_values.reshape(X.shape) 
# Create a contour plot

marg_y = np.trapezoid(F, x, axis=1)   # integrates over x1 values, axis = 0 for going across column for values
marg_x = np.trapezoid(F, y, axis=0 )   # axis = 1 for integrate across each row, leaving you with one result per row

# plt.contour(x, y, F)
# Set plot attributes 

y_values = np.arange(-4, 5, 2)   # gives -4, -2, 0, 2, 4plt.plot(x, trace1, color = "purple", label="Trace 1")
plt.figure(figsize=(7,4.5))

for j in y_values:
    points = np.column_stack((x, np.full_like(x, j)))  # 2D array of x values corresponding to the fixed j values each loop
    z_slice = mvn.pdf(points) # creates new heights for the values that are required
    plt.plot(x, z_slice, label=f"y = {j}", linewidth=2)

plt.xlabel('X') 
plt.ylabel('Y') 
plt.legend(title='Slices at different Y values')
plt.grid(True) 
plt.title('Traces for CMPE 677, Hwk 1, Problem 12', fontsize=12) 
# Save the plot as a PNG file 
plt.savefig('cmpe677_hwk12.png', format='png') 
# Show the plot 
plt.show()