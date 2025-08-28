import numpy as np

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

d = np.array([1,2,3])
f = np. zeros((2,1))
#a) print(B – np.matlib.repmat(f,1,3)) invalid
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