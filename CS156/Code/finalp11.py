x = [[1,0],[0,1],[0,-1],[-1,0],[0,2],[0,-2],[-2,0]]

z1 = [x[i][1]**2-2*x[i][0]-1 for i in range(len(x))]
z2 = [x[i][0]**2-2*x[i][1]+1 for i in range(len(x))]

print(z1,z2)