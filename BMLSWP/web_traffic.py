import scipy as sp
import matplotlib.pyplot as plt

# Get data from the file
data = sp.genfromtxt("web_traffic.tsv", delimiter="\t")

#Preprocess and clean the data
x = data[:,0]
y = data[:,1]

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

#Taking into consideration the inflection between week 3 and week 4
inflection = int(3.5*24*7) #Calculate the inflection point in hours
xa = x[:inflection]
ya = y[:inflection]
xb = x[inflection:]
yb = y[inflection:]

#Error function to calculate the approximation error
def error(f, x, y):
    return sp.sum((f(x) - y)**2)

#Assuming the model is a straight line
fp1, residuals, rank, sc, rcond = sp.polyfit(x,y,1,full=True) #Obtain model parameters
f1 = sp.poly1d(fp1) #This gives us the model function from the model parameters

print (error(f1, x, y))

#straight line model on inflected data
fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)

print(fa_error + fb_error)

#Assuming the model is a polynomial of degree 2
f2p = sp.polyfit(x,y,2)
f2 = sp.poly1d(f2p)

print (error(f2, x, y))

#Degree 10
f10p = sp.polyfit(x,y,10)
f10 = sp.poly1d(f10p)

print (error(f10, x, y))

#Degree 100
f100p = sp.polyfit(x,y,100)
f100 = sp.poly1d(f100p)

print (error(f100, x, y))

#Plotting the chart
fx = sp.linspace(0, x[-1], 1000) #generate X values for plotting
plt.plot(fx, f1(fx), linewidth=2)
plt.plot(fx, f2(fx), linewidth=3)
plt.plot(fx, f10(fx), linewidth=4)
plt.plot(fx, f100(fx), linewidth=5)
plt.plot(fx, fa(fx), linewidth=2)
plt.plot(fx, fb(fx), linewidth=2)
plt.legend(["d=%i" %f1.order], loc="upper left")
plt.scatter(x,y)
plt.title("Web Traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/Hour")
plt.xticks([w*24*7 for w in range(10)], ['week %i'%w for w in range(10)])
plt.autoscale(tight=True)
plt.grid()
plt.show()
