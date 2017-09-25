import numpy as np
import random

coins = np.zeros([1000, 10])
v1 = 0
vrand = 0
vmin = 0
n = 100000

for i in range(n):
	
	for a in range(1000):
		coins[a] = np.random.randint(0,2,size=10) #vector representing 10 coin flips
	
	heads = np.sum(coins, axis=1)
	
	v1 += heads[0]
	vrand += heads[np.random.randint(0,999)]
	vmin += np.amin(heads)
	
print(v1/n/10)
print(vrand/n/10)
print(vmin/n/10)