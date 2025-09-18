import matplotlib.pyplot as plt
import numpy as np
x=[3, -2, 5, 1, 0]
l0 = 0
l1 = 0
l2 = 0
l_inf = 0
for i in x:
    if i != 0:
        l0 = l0 + i
    l1+= (abs(i))
    l2 += (abs(i)**2)
l_inf = max(x)

print("L0 norm is:", l0) 
print("L1 norm is:", l1)
print("L2 norm is:", l2)
print("L infinity norm is:", l_inf)