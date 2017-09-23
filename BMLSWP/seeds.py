import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

data = sp.genfromtxt("seeds.tsv", delimiter="\t")

x = data[:,0]
y = data[:,2]

plt.scatter(x,y)
plt.show()
