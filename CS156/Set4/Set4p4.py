import numpy as np
import random

def linreg(x,y): #function to perform linear regression
	return np.dot(np.linalg.pinv(x),y)
	
w = 0
var = np.zeros([1000])

for i in range(1000): #find w by iterating through a random set of 2 points
	x = 2*np.random.random_sample(2)-1
	xt = np.transpose(np.atleast_2d(x))
	y = np.sin(np.pi*xt)
	var[i] = linreg(xt,y)
	w += linreg(xt,y)
	
w = w/1000
	
print(w)

bias = 0
variance = 0

x1 = np.linspace(-1,1,num=1000)

for i in range(1000): #compare the average w to g
	bias += (np.sin(np.pi*x1[i])-w[0,0]*x1[i])**2

print(bias/1000)

for i in range(1000): #find variance by comparing each of the previous w values to the average
	for j in range(1000):
		variance += (w[0,0]*x1[j]-x1[j]*var[i])**2

print(variance/(1000**2))