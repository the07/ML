import numpy as np

#define a array
a = np.array([0,1,2,3,4,5])

#to know the dimension
a.ndim

#to know the shape
a.shape

b = a.reshape((3,2))

#avoids copies wherever possible so b is just 'a' with different dimension and shape
b[1][0] = 77
#this will make changes to both a and b

#for true copy, c and a are totally independent copies
c = a.reshape((3,2)).copy()

s
