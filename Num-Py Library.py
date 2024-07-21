import numpy as np
from numpy import shape

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# What is NumPy ?
'''
• NumPy stands for Numerical Python and is an excellent way to deal with arrays and just large amount of data in general 
  within Python.
'''

# Why use Num-Py ?
'''
• NumPy provides effiecient storage.
• It also provides better ways of handling data for processing.
• It is easy to learn
• NumPy uses relatively less memory to store data
'''

# NumPy vs List
'''
• It occupies less memory compared to List.
• It is pretty fast.
• It is very convenient to work with NumPy.
'''

# Application of NumPy
'''
• Mathematics(MATLAB Replacement)
• Plotting(Matplotlib)
• Backend(Pandas, Connect 4, Digital Photography)
'''

# EXAMPLE: 1-D ARRAY
'''
np1 = np.array([1, 2, 3, 4, 5])          # check for 'int16', 'int64'
print(np1)
'''
# EXAMPLE: 2-D ARRAY
'''
np2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(np2)
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# ARRAY CREATION
'''
1. Conversion from other Python structures (i.e. lists and tuples)
2. Intrinsic NumPy array creation functions (e.g. arange, ones, zeros, etc.)
'''

# ARRAY CREATION: 1. Conversion from other Python Structures (i.e. lists and tuples)
'''
• A list of numbers will create a 1-D array.
• A list of list will create a 2-D array.
• Further nested list will create higher-dimensional arrays.
• In general, any array object is called an "ndarray" in NumPy.
'''

# EXAMPLE 1:
'''
ListArray1 = np.array([1, 2, 3, 4, 5])                                  # 1-D array
print(ListArray1)

ListArray2 = np.array([[1, 2], [3, 4], [5, 6]])                         # 2-D array
print(ListArray2)

ListArray3 = np.array([[[1, 2, 3], [11, 12, 13], [21, 22, 23]]])        # 3-D array
print(ListArray3)

TupleArray = np.array([(1, 2), (3, 4), (5, 6)])
print(TupleArray)

TupleArray2 = np.array(((1, 2), (3, 4), (5, 6)))
print(TupleArray2)

TupleArray3 = np.array((((1, 2), (3, 4), (5, 6))))
print(TupleArray3)
'''

# EXAMPLE 2: MEMORY MANAGEMENT
'''
np3 = np.array([1, 3, 5, 7, 9], np.int8)
print(np3)

np4 = np.array([2, 4, 6, 8, 1000], np.int8)         # problem occured
print(np4)

np5 = np.array([2, 4, 6, 8, 1000], np.int16)
print(np5)

np6 = np.array([3, 6, 9, 12, 5555555555], np.int32)         # OverflowError: int too large
print(np6)

np7 = np.array ([3, 6, 9, 12, 5555555555], dtype='int64')
print(np7)
'''

# ARRAY CREATION: 2. Intrinsic NumPy array creation functions (e.g. arange, ones, zeros, etc.)

# 2.1: 1-D ARRAY CREATION FUNCTION
'''
• The 1-D array creation function e.g. "numpy.linspace" and "numpy.arange"
• numpy.arange creates arrays with regularly incrementing value.
• numpy.linspace will create array with a specified number of elements and spaced equally between the specified beginning
  and end values.
'''

# EXAMPLE: numpy.arange
'''
Arange = np.arange(10)
print(Arange)

Arange1 = np.arange(-10)
print(Arange1)
'''

# EXAMPLE: numpy.linspace
'''
Linspace = np.linspace(1, 4, 4)
print(Linspace)

Linspace1 = np.linspace(1, 5, 4)
print(Linspace1)
'''

# 2.2: 2-D ARRAY CREATION FUNCTION
'''
• The 2-D array creation function e.g. numpy.eye, numpy.diag, numpy.vander
• numpy.eye defines a 2-D identity matrix
• numpy.diag can define either a square 2D array with given values along the diagonal elemnts.
• numpy.vander function is used to generate Vandermonde matrix.
'''

# EXAMPLE: nump.eye SIMILAR TO numpy.identity
'''
Eye = np.eye(3)
print(Eye)

Eye1 = np.eye(5, 5)
print(Eye1)

Identity = np.identity(3)
print(Identity)

Identity1 = np.identity(5)
print(Identity1)
'''

# EXAMPLE: numpy.diag
'''
Diagonally = np.diag([1, 2, 3])
print(Diagonally)

Diagonally1 = np.diag([1, 2, 3], 1)
print(Diagonally1)
'''

# EXAMPLE: numpy.vander

# Eg.1:
'''
• N is not given therefore N = len(input matrix) i.e N = len(X)
• Power = N-1 = 4-1 = 3
• increasing = False

X = np.vander([1, 2, 3, 4])
print(X)
'''
# Eg.2:
'''
• N is given 5
• therefore, Power = N-1 = 5-1 = 4

X = np.vander([1, 2, 3, 4, 5], 5, increasing=True)
print(X)
'''
# Eg.3:
'''
Vandermonde = np.array([1, 2, 3])
X = np.vander(Vandermonde, 5, increasing=True)
print(X)
'''

# 2.3: GENERAL NDARRAY CREATION FUNCTION
'''
• The ndarray creation function e.g. numpy.ones, numpy.zeros and random
• numpy.zeros wiil create an array filled with 0 values with specified shape.
• numpy.ones will create an array filed with 1 value. It is identical to zeros in all other respects.
• The random method of the default_rng will create as array filled with random value between 0 and 1.
• numpy.indices will create a set of arrays, one per dimension with each representing variation in that dimension.
'''

from numpy.random import default_rng

# EXAMPLE: numpy.zeros
'''
Zero = np.zeros((2, 3))
print(Zero)

Zero1 = np.zeros((2, 3, 2))
print(Zero1)
'''

# EXAMPLE: numpy.ones
'''
One = np.ones((2, 3))
print(One)

One1 = np.ones((2, 3, 2))
print(One1)
'''

# EXAMPLE: random
'''
Random = default_rng(42).random((2, 3))
print(Random)

Random1 = default_rng(42).random((2, 3, 2))
print(Random1)
'''

# EXAMPLE: numpy.indices
'''
Indices = np.indices((3,1))
print(Indices)
print("•••••••••••••••")
Indices1 = np.indices((3,2))
print(Indices1)
print("•••••••••••••••")
Indices2 = np.indices((3,3))
print(Indices2)
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# ANOTHER TYPES OF ARRAY

# EXAMPLE: BLOCK
'''
A = np.zeros((2, 2))
B = np.ones((2,2))
C = np.eye(2,2)
D = np.diag((-5,-2))

Block = np.block([[A, B], [C, D]])
print(Block)
'''

# EXAMPLE: FULL
'''
Full = np.full(A.shape,5)
print(Full)

Full_Like = np.full_like(A,7)
print(Full_Like)

Full1 = np.full((3,2), 5)
print(Full1)
'''

# EXAMPLE: RANDOM DECIMAL NUMBERS
'''
Rdn = np.random.rand(4,2)
print(Rdn)

Rdn1 = np.random.rand(4,2,3)
print(Rdn1)

Rdn2 = np.random.random_sample(A.shape)
print(Rdn2)

Rdn3 = np.random.randint(5, size=(3,3))
print(Rdn3)

Rdn4 = np.random.random_integers(1,9,(3,3))         # Warning
print(Rdn4)
'''

# EXAMPLE: REPEAT AN ARRAY
'''
Rep = np.array([[1, 2, 3]])
Repeat = np.repeat(Rep,3, axis=0)
print(Repeat)
'''

# EXAMPLE: CREATING CUSTOM ARRAY
'''
o = np.ones((5,5))
z = np.zeros((3,3))

z[1,1] = 5
o[1:4,1:4] = z

print(o)
'''

# EXAMPLE: COPYING ARRAYS
'''
a = np.array([1, 2, 3, 4, 5])
b = a
b[0] = 100
print(b)
print(a)
'''

'''
a = np.array([1, 2, 3, 4, 5])
b = a.copy()
b[0] = 100
print(b)
print(a)
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# EXAMPLE: SLICING OF 1-D ARRAY
'''
np8 = np.array([1, 2, 3])           # 1-D array
print(np8)

print(np8[0])
print(np8[0,0])             # IndexError
'''

# EXAMPLE: SLICING OF 2-D ARRAY
'''
np9 = np.array([[1, 2, 3, 4, 16, 32, 64, 128, 256], [4, 5, 6, 7, 49, 98, 196, 392, 784], [7, 8, 9, 0, 1, 2, 4, 8, 16]])        # 2-D array
print(np9)

print(np9[0])
print(np9[1])
print(np9[0,0])
print(np9[0,1])
print(np9[1,1]

print(np9[0,:])                 # Accessing 0th row
print(np9[2,:])                 # Accessing 2nd row

print(np9[:,2])                 # Accessing 2nd column
print(np9[:,5])                 # Accessing 5th column

print(np9[0, 1:7:1])            # Accessing specific range from 0th row
print(np9[1, 1:8:2])            # Accessing specific elements from 1st row

np9[1, 2] = 36                  # Changing specific element
print(np9)

np9[:,2] = 555                  # Changing specific column value
print(np9)
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# NUMPY AXIS
'''
• Axes are defind for arrays with more than one dimension.
• A 2-dimensional array has two corresponding axes: the first running vertically downwards across rows (axis 0) and the
  second running horizontally across columns (axis 1)
'''

# EXAMPLE:
'''
Axis_array = np.array([[3, 2, 1], [5, 6, 4], [7, 8, 9]])
print(Axis_array)

Ax0 = Axis_array.sum(axis=0)
print(Ax0)

Ax1 = Axis_array.sum(axis=1)
print(Ax1)
'''

# ARRAY METHODS
'''
print(type(Axis_array))         # Type of Array
print(Axis_array.dtype)         # Datatype
print(Axis_array.size)          # No. of Elements
print(Axis_array.shape)         # Matrix Size
print(Axis_array.ndim)          # No. of Dimensions
print(Axis_array.itemsize)
print(Axis_array.nbytes)        # Total bytes consumed
print(Axis_array.T)             # Transpose of Matrics

for i in Axis_array.flat:
        print(i)
'''

# REORGANIZING ARRAYS
'''
a = np.arange(10)
print(a)

b = a.reshape(2, 5)
print(b)

c = a.reshape(5, 2)
print(c)

Ravel = b.ravel()
print(Ravel)
'''

# ARRAY Functions
'''
ONE_Dim = np.array([5, 10, 55555, 20, 25])
print(ONE_Dim)

print(ONE_Dim.argmax())         # shows index no. where max element is appeared
print(ONE_Dim.argmin())         # shows index no. where min element is appeared
print(ONE_Dim.argsort())        # sort the element in ascending order, showing with index no.
'''

# ARRAY Functions
'''
TWO_Dim = np.array([[2, 6, 4], [9, 6, 3], [4, 8, 12]])
# print(TWO_Dim)

print(TWO_Dim.argmax())         # first sort the element in ascending order and then argmax
print(TWO_Dim.argmin())         # first sort the element in ascending order and then argmin
print(TWO_Dim.argsort())
print(TWO_Dim.argmax(axis=0))
print(TWO_Dim.argmax(axis=1))
print(TWO_Dim.argmin(axis=0))
print(TWO_Dim.argmin(axis=1))
print(TWO_Dim.argsort(axis=0))
print(TWO_Dim.argsort(axis=1))
print(TWO_Dim.ravel())
print(TWO_Dim.reshape(3, 3))
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# ARITHMETIC OPERATIONS

A1 = np.array([[1, 2, 1], [1, 4, 4], [4, 6, 9]])
'''
A2 = np.array([[0, 1, 2], [1, 2, 3], [0, 1, 0]])

print(A1+A2)
print(A1-A2)
print(A1*A2)
print(A1/A2)

print(A1+2)
print(A1-2)
print(A1*2)
print(A1/2)
print(A1**2)
print(A1//2)

Square_Root = np.sqrt(A1)
print(Square_Root)

Square = np.square(A1)
print(Square)

print(A1.sum())
print(A1.max())
print(A1.min())
print(A1.mean())
print(A1.std())

print(np.sin(A1))
print(np.cos(A1))
print(np.tan(A1))

print(np.exp(A1))

print(np.log(A1))           # Natural Log i.e. ln
print(np.log10(A1))         # Log to the base 10
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# LINEAR ALGEBRA

# EXAMPLE: 1. Matmul
'''
a = np.full((3,2), 2)
print(a)
b = np.ones((2,3))
print(b)

X = np.matmul(a,b)
print(X)
'''

# EXAMPLE: 2. Matmul
'''
a = np.full((4, 2), 5)
print(a)
b = np.ones((2,4))
print(b)

X = np.matmul(a,b)
print(X)
'''

# Determinant
'''
a = np.identity(3)
X = np.linalg.det(a)
print(X)

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 1]])
X = np.linalg.det(b)
print(X)
'''

# Statistics
'''
Stats = np.array([[1, 5, 3], [4, 2, 6]])
print(Stats)

a = np.min(Stats)
print(a)
a1 = np.min(Stats, axis=0)
print(a1)
a2 = np.min(Stats, axis=1)
print(a2)

b = np.max(Stats)
print(b)
b1 = np.max(Stats, axis=0)
print(b1)
b2 = np.max(Stats, axis=1)
print(b2)

c = np.sum(Stats)
print(c)
c1 = np.sum(Stats, axis=0)
print(c1)
c2 = np.sum(Stats, axis=1)
print(c2)

d = np.mean(Stats)
print(d)
d1 = np.mean(Stats, axis=0)
print(d1)
d2 = np.mean(Stats, axis=1)
print(d2)
'''

# Vertically Stacking Vector
'''
V1 = np.array([1, 2, 3, 4, 5])
V2 = np.array([6, 7, 8, 9, 0])

VStack = np.vstack((V1,V2))
print(VStack)

VStack1 = np.vstack((V1,V2,V1,V2))
print(VStack1)
'''

# Horizontally Stacking Vector
'''
H1 = np.array([1, 2, 3, 4, 5])
H2 = np.array([6, 7, 8, 9, 0])

HStack = np.hstack((H1,H2))
print(HStack)

HStack1 = np.hstack((H1,H2,H1,H2))
print(HStack1)
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# Counting Non_Zero Elements and its Position
'''
Non_Zero = np.count_nonzero(A1)
print(Non_Zero)

Nonzero_Position = np.nonzero(A1)
print(Nonzero_Position)

A1[2, 2] = 0
print(np.nonzero(A1))
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# LOAD DATA FROM FILE - .txt only
'''
Data = np.genfromtxt('NumPy_Test.txt', delimiter=',')

print(Data)                         # in float

print(Data.astype('int32'))         # in integer
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# BOOLEAN MASKING & ADNANCED INDEXING

# Boolean Masking
'''
print(Data > 25)

x = np.any(Data > 25)
print(x)
x1 = np.any(Data > 25, axis=0)
print(x1)
x2 = np.any(Data > 25, axis=1)
print(x2)

print((Data > 20) & (Data < 50))
print(~((Data > 20) & (Data < 50)))
'''

# Advanced Indexing
'''
AI = np.array([5, 25, 4, 16, 3, 9, 2, 4])

print(AI[[0,1]])
print(AI[[6,7]])
print(AI[[1, 0, 5, 7]])
'''

# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•

# SINE and COSIN GRAPHS

import matplotlib.pyplot as plt

# SIN graph
'''
x = np.arange(0, 3*np.pi, 0.1)
y = np.sin(x)

plt.plot(x,y)

plt.show()
'''

# COS graph
'''
x = np.arange(0, 3*np.pi, 0.1)
y = np.cos(x)

plt.plot(x,y)

plt.show()
'''

# TAN graph
'''
x = np.arange(0, 3*np.pi, 0.1)
y = np.tan(x)

plt.plot(x,y)

plt.show()
'''


# ○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•○•