'''
Train a Q-value following a classical Q-learning algorithm (enforcing the
satisfaction of HJB method), using a noisy greedy exploration strategy.

The result of a training for a continuous pendulum (after 200 iterations) 
are stored in qvalue.h5.

Reference:
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." 
Nature 518.7540 (2015): 529.
'''

from tp7.env_pendulum import EnvPendulumHybrid; Env = lambda : EnvPendulumHybrid(1,viewer='meshcat')
from tp7.qnetwork import QNetwork
from collections import deque
import time
import signal
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf

### --- Random seed
RANDOM_SEED = int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)

### --- Environment
env                 = Env()

### --- Hyper paramaters
NEPISODES               = 1000          # Max training steps
NSTEPS                  = 60            # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 32            # Hidden layer size

### --- Replay memory
class ReplayItem:
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2
replayDeque = deque()

### --- Tensor flow initialization
qvalue          = QNetwork(nx=env.nx,nu=env.nu,learning_rate=QVALUE_LEARNING_RATE)
qvalueTarget    = QNetwork(name='target',nx=env.nx,nu=env.nu)
# Uncomment to load networks
#qvalue.load()
#qvalueTarget.load()

def rendertrial(maxiter=NSTEPS,verbose=True):
    x = env.reset()
    traj = [x.copy()]
    rsum = 0.
    for i in range(maxiter):
        u = qvalue.policy(x)[0]
        x, reward = env.step(u)
        env.render()
        time.sleep(1e-2)
        rsum += reward
        traj.append(x.copy())
    if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
    return np.array(traj)
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed

### History of search
h_rwd = []

### --- Training
for episode in range(1,NEPISODES):
    x    = env.reset()
    rsum = 0.0

    for step in range(NSTEPS):
        u       = qvalue.policy(x,                                     # Greedy policy ...
                                noise=1. / (1. + episode + step))      # ... with noise
        x2,r    = env.step(u)
        done    = False # Some environment may return information when task completed

        replayDeque.append(ReplayItem(x,u,r,done,x2))                # Feed replay memory ...
        if len(replayDeque)>REPLAY_SIZE: replayDeque.popleft()       # ... with FIFO forgetting.

        rsum   += r
        x       = x2
        if done: break
        
        # Start optimizing networks when memory size > batch size.
        if len(replayDeque) > BATCH_SIZE:     
            batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
            x_batch    = np.vstack([ b.x      for b in batch ])
            u_batch    = np.vstack([ b.u      for b in batch ])
            r_batch    = np.array([ [b.reward] for b in batch ])
            d_batch    = np.array([ [b.done]   for b in batch ])
            x2_batch   = np.vstack([ b.x2     for b in batch ])
            
            # Compute Q(x,u) from target network
            v_batch    = qvalueTarget.value(x2_batch)
            qref_batch = r_batch + (d_batch==False)*(DECAY_RATE*v_batch)

            # Update qvalue to solve HJB constraint: q = r + q'
            qvalue.trainer.train_on_batch([x_batch,u_batch],qref_batch)
            
            # Update target networks by homotopy.
            qvalueTarget.targetAssign(qvalue,UPDATE_RATE)
      
    # \\\END_FOR step in range(NSTEPS)

    # Display and logging (not mandatory).
    print('Ep#{:3d}: lasted {:d} steps, reward={:3.0f}' .format(episode, step,rsum))
    h_rwd.append(rsum)
    if not (episode+1) % 200:     rendertrial(30)

# \\\END_FOR episode in range(NEPISODES)

print("Average reward during trials: %.3f" % (sum(h_rwd)/NEPISODES))
rendertrial()
plt.plot( np.cumsum(h_rwd)/range(1,NEPISODES) )
plt.show()

# Uncomment to save networks
#qvalue.save()
