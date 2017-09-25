import numpy as np

count1 = 0
count2 = 0
count3 = 0

for i in range(1000):
	x1 = np.random.uniform()
	x2 = np.random.uniform()
	x3 = min(x1,x2)
	count1 += x1
	count2 += x2
	count3 += x3

print(count1/1000, count2/1000, count3/1000)