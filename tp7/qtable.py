'''
Example of Q-table learning with a simple discretized 1-pendulum environment.
-- concerge in 1k  episods with pendulum(1)
-- Converge in 10k episods with cozmo model
'''

import matplotlib.pyplot as plt
import signal
import time
import numpy as np

### --- Random seed
RANDOM_SEED = 1188 #int((time.time()%10)*1000)
print("Seed = %d" % RANDOM_SEED)
np.random.seed(RANDOM_SEED)

### --- Environment
from tp6.env_pendulum import EnvPendulumDiscrete; Env = lambda : EnvPendulumDiscrete(1,viewer='meshcat')
env = Env()

### --- Hyper paramaters
NEPISODES               = 400           # Number of training episodes
NSTEPS                  = 50            # Max episode length
LEARNING_RATE           = 0.85          # 
DECAY_RATE              = 0.99          # Discount factor 

Q     = np.zeros([env.nx,env.nu])       # Q-table initialized to 0

def policy(x):
    return np.argmax(Q[x,:])

def rendertrial(s0=None,maxiter=100):
    '''Roll-out from random state using greedy policy.'''
    s = env.reset(s0)
    traj = [s]
    for i in range(maxiter):
        a = np.argmax(Q[s,:])
        s,r = env.step(a)
        env.render()
        traj.append(s)
    return traj
    
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

h_rwd = []                              # Learning history (for plot).
for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0
    for steps in range(NSTEPS):
        u         = np.argmax(Q[x,:] + np.random.randn(1,env.nu)/episode) # Greedy action with noise
        x2,reward = env.step(u)
        
        # Compute reference Q-value at state x respecting HJB
        Qref = reward + DECAY_RATE*np.max(Q[x2,:])

        # Update Q-Table to better fit HJB
        Q[x,u] += LEARNING_RATE*(Qref-Q[x,u])
        x       = x2
        rsum   += reward

    h_rwd.append(rsum)
    if not episode%20:
        print('Episode #%d done with average cost %.2f' % (episode,sum(h_rwd[-20:])/20))

print("Total rate of success: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()

