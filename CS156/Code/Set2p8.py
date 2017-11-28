import numpy as np
import random

misc = 0
n = 1000

for t in range(1000):
	
	points = np.zeros([n, 3])
	y = np.zeros([n])
	ytrue = np.zeros([n])
	
	for i in range(n): #same code as before
		points[i,0] = 1 #set all x0 to 1
		points[i,1] = np.random.uniform(-1,1) #set all x1 to the x-coordinate of point
		points[i,2] = np.random.uniform(-1,1) #set all x2 to the y-coordinate of point
		ytrue[i] = np.sign(np.square(points[i,1]) + np.square(points[i,2]) - 0.6)
		if np.random.randint(0,10) != 1:
			y[i] = np.sign(np.square(points[i,1]) + np.square(points[i,2]) - 0.6)
		else:
			y[i] = -1*np.sign(np.square(points[i,1]) + np.square(points[i,2]) - 0.6)
		
	w = np.dot(np.dot(np.linalg.inv(np.dot(np.matrix.transpose(points),points)),np.matrix.transpose(points)),y)
	g = np.sign(np.dot(points,w))
	
	misc += (1000-np.sum(np.dot(g,ytrue)))/2

print(misc/1000000)