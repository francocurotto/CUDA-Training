# imports
import numpy as np
import matplotlib.pyplot as plt

# get data
data = np.loadtxt("compare.txt",delimiter=",");
nthreads = data[:,0]
nblocks = data[:,1]
time = data[:,2]
nelems = nblocks[0]

# plot
#plt.loglog(nthreads, time, "-o")
plt.semilogx(nthreads, time, "-o")
plt.grid()
plt.xlabel("n threads")
plt.ylabel("time [s]")
# second x axis
f = lambda x:nelems/x
ax2 = plt.gca().secondary_xaxis("top", functions=(f, f))
ax2.set_xlabel("n blocks")

# show
plt.show()
