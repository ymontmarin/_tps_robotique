import matplotlib.pylab as plt
import numpy as np

def plotFlow(env,policy,x2d):
    flow = []
    for s in range(env.nx):
        env.reset(s)
        x = x2d(s)
        a = policy(s)
        snext,r = env.step(a)
        xnext = x2d(snext)
        flow.append( [x,xnext-x] )

    flow=np.array( [ np.concatenate(a) for a in flow ])
    h = plt.quiver(flow[:,0],flow[:,1],flow[:,2],flow[:,3])
