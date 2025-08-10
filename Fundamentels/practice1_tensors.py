import torch
import time

# # SCALAR
# #A scalar is a single number and in tensor-speak it's a zero dimension tensor.

# scalar = torch.tensor(7)
# print(scalar.item) # scalar.item() returns the value of the scalar tensor
# print(scalar.ndim) # scalar.ndim returns the number of dimensions of the tensor, which is 0 for a scalar

# #VECTORS
# # A vector is a one-dimensional tensor, which can be thought of as a list of numbers.

# vector = torch.tensor([1, 2, 3, 4, 5])
# print(vector)  # prints the vector
# print(vector.ndim)  # prints the number of dimensions, which is 1
# print(vector.shape)  # prints the shape of the vector, which is (5,) indicating 5 elements in one dimension

# # MATRICES
# # A matrix is a 2-dimensional tensor

# matrix = torch.tensor([[1, 2, 3], 
#                        [4, 5, 6]])
# print(matrix.size())  # prints the size of the matrix, which is (2, 3) indicating 2 rows and 3 columns
# print(matrix.shape)  # prints the shape of the matrix, which is also (2,3)

# # TENSOR
# # A tensor is a multi-dimensional array

# TENSOR = torch.tensor([[[1, 2, 3],
#                         [3, 6, 9],
#                         [2, 4, 5]]])
# print(TENSOR.shape)  # prints the shape of the tensor, which is also (1, 3, 3) where 1 is the number of matrices, 3 is the number of rows, and 3 is the number of columns


# CREATING RANDOM TENSORS
random_tensor = torch.rand(size=(3, 4))  # creates a random tensor of shape (3, 4)
print(random_tensor.shape) 
print(random_tensor)    # prints the random tensor. Ex: tensor([[0.3789, 0.7876, 0.9350, 0.3086],
                                                                #[0.1179, 0.8138, 0.0677, 0.6266],
                                                                #[0.5102, 0.1893, 0.3975, 0.9887]])

# CREATING A ZERO TENSOR
zero_tensor = torch.zeros(size=(2, 3))  # creates a zero tensor with all values of 0 and 2 rows and 3 columns
ones_tensor = torch.ones(size=(2, 3))  # creates a ones tensor with all values of 1 and 2 rows and 3 columns
print(ones_tensor.dtype) # data type of tensor, which is torch.float32 = 32 # bit floating point number

# CREATING A RANGE TENSOR
range_tensor = torch.arange(start=0, end=10, step=2)  # creates a tensor with values from 0 to 10 with a step of 2
print(range_tensor)  # prints the range tensor, which is tensor([0, 2, 4, 6, 8])

# DATA TYPES OF TENSORS

float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded 

print(float_32_tensor)
print(f"Shape of tensor: {float_32_tensor.shape}")
print(f"Datatype of tensor: {float_32_tensor.dtype}")
print(f"Device tensor is stored on: {float_32_tensor.device}")

# For calculation purposes, tensors should be of the same data type and on the same device.

# If you want to change the data type of a tensor, you can use the .to() method
float_64_tensor = float_32_tensor.to(torch.float64)  # converts the tensor to float64 data type

#-------------------------------------------------------------------------------------------------------------------------


#                                                    OPERATIONS ON TENSORS

# Addition, subtraction, multiplication, and division can be performed on tensors.

# Addition
tensor_a = torch.tensor([1, 2, 3])
print(tensor_a + 2)  # adds 2 to each element of the tensor, resulting in tensor([3, 4, 5])

# Multiplication
tensor_b = torch.tensor([4, 5, 6])
torch.multiply(tensor_b, 10) # multiplies each element of the tensor by 10, resulting in tensor([10, 20, 30])

# Element-wise multiplication (each element multiplies its equivalent, index 0->0, 1->1, 2->2)

print(tensor_a, "*", tensor_b)
print("Equals:", tensor_a * tensor_b)


# MATRIX MULTIPLICATION
# Matrix multiplication can be performed using the @ operator or torch.matmul() function.
# (2, 3) @ (3, 2) -> (2, 2)

#       Operation	                                Calculation	                                Code
#       Element-wise multiplication	                [1*1, 2*2, 3*3] = [1, 4, 9]	            tensor * tensor
#       Matrix multiplication	                    [1*1 + 2*2 + 3*3] = [14]	            tensor.matmul(tensor)


tensor_mul = torch.tensor([1, 2, 3])
tensor_mul2 = torch.tensor([4, 5, 6])
print(tensor_mul @ tensor_mul2)  # performs matrix multiplication, resulting in tensor(32) which is 1*4 + 2*5 + 3*6 = 32
print(torch.matmul(tensor_mul, tensor_mul2))  # same as above, resulting in tensor(32)


#                               SHOWING THE TIME TAKEN FOR MATRIX MULTIPLICATION BY HAND
start = time.time()
# Matrix multiplication by hand 
# (avoid doing operations with for loops at all cost, they are computationally expensive)
tensor = tensor_a
value = 0
for i in range(len(tensor)):
  value += tensor[i] * tensor[i]
value

end = time.time()
print(f"Time taken for manual multiplication: {end - start} seconds")


# COMMON ERRORS IN DEEP LEARNING: Shape errors
# Shape errors occur when the dimensions of the tensors do not match for the operation being performed.

# Shapes need to be in the right way  
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)

# torch.matmul(tensor_A, tensor_B) 
# # (this will error) since tensor_A is (3, 2) and tensor_B is (3, 2), they cannot be multiplied directly.
# To fix this, we can transpose tensor_B to make it (2, 3) so that the multiplication can be performed.
tensor_B = tensor_B.T  # transposes tensor_B to (2, 3)
print(torch.matmul(tensor_A, tensor_B))  # now it works, resulting in a tensor of shape (3, 3)

# _____________________________________________________________________________________________________________________________________________
# torch.mm(tensor_A, tensor_B)  # is the shoryform for torch.matmul()

# MANUAL SEEDING FUNCTION
# # Setting a manual seed ensures that the random numbers are generated when using RGN (Random Number Generator)


        # # torch.manual_seed() only controls the RNG for PyTorch CPU and (by default) CUDA.
        # torch.manual_seed(42)  # Set the seed
        # print(torch.rand(2, 2))  # Always the same output in the terminal after the seed was set. We can set the seed again to reset it

        # # the number in torch.manual_seed() whenever called will always produce the same random numbers.
        # # this is for reproducibility, so that the same random numbers are generated every time the code is run.

        # torch.manual_seed(7)  # gives another value to seed 7's random generator
        # print(torch.rand(2, 3)) # different output than seed 42

        # torch.manual_seed(42)  # Reset the seed again
        # print(torch.rand(2, 2)) # gives same output as on line 141
# _____________________________________________________________________________________________________________________________________________

torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input 
                         out_features=6) # out_features = describes outer value 
x = tensor_A
output = linear(x)
print(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")


#______________________________________________________________________________________________________________________________________________

# SOME AGGREGATE METHODS: max, min, mean, sum, std, var

x = torch.arange(1, 100, 10)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
                                    # print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}")      # won't work without float datatype
print(f"Sum: {x.sum()}")

