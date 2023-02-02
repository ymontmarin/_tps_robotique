'''
Deep actor-critic network, 
From "Continuous control with deep reinforcement learning", by Lillicrap et al, arXiv:1509.02971
'''

from env_pendulum import EnvPendulumSinCos; Env = lambda : EnvPendulumSinCos(1,viewer='meshcat')
import gym
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from collections import deque
import signal

#######################################################################################################33
#######################################################################################################33
#######################################################################################################33
### --- Random seed
RANDOM_SEED = 0 # int((time.time()%10)*1000)
print("Seed = %d" %  RANDOM_SEED)
np .random.seed     (RANDOM_SEED)
random.seed         (RANDOM_SEED)
tf.random.set_seed  (RANDOM_SEED)

### --- Hyper paramaters
NEPISODES               = 1000           # Max training steps
NSTEPS                  = 200           # Max episode length
QVALUE_LEARNING_RATE    = 0.001         # Base learning rate for the Q-value Network
POLICY_LEARNING_RATE    = 0.0001        # Base learning rate for the policy network
DECAY_RATE              = 0.99          # Discount factor 
UPDATE_RATE             = 0.01          # Homotopy rate to update the networks
REPLAY_SIZE             = 10000         # Size of replay buffer
BATCH_SIZE              = 64            # Number of points to be fed in stochastic gradient
NH1 = NH2               = 250           # Hidden layer size
EXPLORATION_NOISE       = 0.2

### --- Environment
# problem = "Pendulum-v1"
# env = gym.make(problem)
# NX = env.observation_space.shape[0]
# NU = env.action_space.shape[0]
# UMAX = env.action_space.high[0]
# env.reset(seed=RANDOM_SEED)
# assert( env.action_space.low[0]==-UMAX)

env                 = Env()             # Continuous pendulum
NX                  = env.nx            # ... training converges with q,qdot with 2x more neurones.
NU                  = env.nu            # Control is dim-1: joint torque
UMAX                = env.umax[0]       # Torque range


#######################################################################################################33
### NETWORKS ##########################################################################################33
#######################################################################################################33

class QValueNetwork:
    '''
    Neural representaion of the Quality function:
    Q:  x,y -> Q(x,u) \in R
    '''
    def __init__(self,nx,nu,nhiden1=32,nhiden2=256,learning_rate=None):

        state_input = tfk.layers.Input(shape=(nx))
        state_out = tfk.layers.Dense(nhiden1, activation="relu")(state_input)
        state_out = tfk.layers.Dense(nhiden1, activation="relu")(state_out)

        action_input = tfk.layers.Input(shape=(nu))
        action_out = tfk.layers.Dense(nhiden1, activation="relu")(action_input)

        concat = tfk.layers.Concatenate()([state_out, action_out])

        out = tfk.layers.Dense(nhiden2, activation="relu")(concat)
        out = tfk.layers.Dense(nhiden2, activation="relu")(out)
        value_output = tfk.layers.Dense(1)(out)

        self.model = tfk.Model([state_input, action_input], value_output)

    @tf.function
    def targetAssign(self,target,tau=UPDATE_RATE):
        for (tar,cur) in zip(target.model.variables,self.model.variables):
            tar.assign(cur * tau + tar * (1 - tau))
 

class PolicyNetwork:
    '''
    Neural representation of the policy function:
    Pi: x -> u=Pi(x) \in R^nu
    '''
    def __init__(self,nx,nu,umax,nhiden=32,learning_rate=None):
        random_init = tf.random_uniform_initializer(minval=-0.005, maxval=0.005)
        
        state_input = tfk.layers.Input(shape=(nx,))
        out = tfk.layers.Dense(nhiden, activation="relu")(state_input)
        out = tfk.layers.Dense(nhiden, activation="relu")(out)
        policy_output = tfk.layers.Dense(1, activation="tanh",
                                         kernel_initializer=random_init)(out)*umax
        self.model = tfk.Model(state_input, policy_output)

    @tf.function
    def targetAssign(self,target,tau=UPDATE_RATE):
        for (tar,cur) in zip(target.model.variables,self.model.variables):
            tar.assign(cur * tau + tar * (1 - tau))

    def numpyPolicy(self,x,noise=None):
        '''Eval the policy with numpy input-output (nx,)->(nu,).'''
        x_tf = tf.expand_dims(tf.convert_to_tensor(x), 0)
        u = np.squeeze(self.model(x_tf).numpy(),0)
        if noise is not None:
            u = np.clip( u+noise, -UMAX,UMAX)
        return u

    def __call__(self, x,**kwargs):
        return self.numpyPolicy(x,**kwargs)

            
        
#######################################################################################################33

class OUNoise:
    '''
    Ornstein–Uhlenbeck processes are markov random walks with the nice property to eventually
    converge to its mean.
    We use it for adding some random search at the begining of the exploration.
    '''
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, y_initial=None,dtype=np.float32):
        self.theta = theta
        self.mean = mean.astype(dtype)
        self.std_dev = std_deviation.astype(dtype)
        self.dt = dt
        self.dtype=dtype
        self.reset(y_initial)

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        noise = np.random.normal(size=self.mean.shape).astype(self.dtype)
        self.y += \
            self.theta * (self.mean - self.y) * self.dt \
            + self.std_dev * np.sqrt(self.dt) * noise
        return self.y.copy()

    def reset(self,y_initial = None):
        self.y = y_initial.astype(self.dtype) if y_initial is not None else np.zeros_like(self.mean)

### --- Replay memory
class ReplayItem:
    '''
    Storage for the minibatch
    '''
    def __init__(self,x,u,r,d,x2):
        self.x          = x
        self.u          = u
        self.reward     = r
        self.done       = d
        self.x2         = x2


#######################################################################################################33
quality = QValueNetwork(NX,NU,NH1,NH2)
qualityTarget = QValueNetwork(NX,NU,NH1,NH2)
quality.targetAssign(qualityTarget,1)

policy = PolicyNetwork(NX,NU,umax=UMAX,nhiden=NH2)
policyTarget = PolicyNetwork(NX,NU,umax=UMAX,nhiden=NH2)
policy.targetAssign(policyTarget,1)

replayDeque = deque()

ou_noise = OUNoise(mean=np.zeros(1), std_deviation=float(EXPLORATION_NOISE) * np.ones(1))
ou_noise.reset( np.array([ UMAX/2 ]) )

#######################################################################################################33
### MAIN ACTOR-CRITIC BLOCK
#######################################################################################################33

critic_optimizer = tfk.optimizers.Adam(QVALUE_LEARNING_RATE)
actor_optimizer = tfk.optimizers.Adam(POLICY_LEARNING_RATE)

@tf.function
def learn(state_batch, action_batch, reward_batch, next_state_batch):
    '''
    <learn> is isolated in a tf.function to make it more efficient.
    @tf.function forces tensorflow to optimize the inner computation graph defined in this function.
    '''

    # Automatic differentiation of the critic loss, using tf.GradientTape
    # The critic loss is the classical Q-learning loss:
    #         loss = || Q(x,u) -  (reward + Q(xnext,Pi(xnexT)) ) ||**2
    with tf.GradientTape() as tape:
        target_actions = policyTarget.model(next_state_batch, training=True)
        y = reward_batch + DECAY_RATE * qualityTarget.model(
            [next_state_batch, target_actions], training=True
        )
        critic_value = quality.model([state_batch, action_batch], training=True)
        critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        
    critic_grad = tape.gradient(critic_loss, quality.model.trainable_variables)
    critic_optimizer.apply_gradients(
        zip(critic_grad, quality.model.trainable_variables)
    )

    # Automatic differentiation of the actor loss, using tf.GradientTape
    # The actor loss implements a greedy optimization on the quality function
    #           loss(u) = Q(x,u)
    with tf.GradientTape() as tape:
        actions = policy.model(state_batch, training=True)
        critic_value = quality.model([state_batch, actions], training=True)
        actor_loss = -tf.math.reduce_mean(critic_value)

    actor_grad = tape.gradient(actor_loss, policy.model.trainable_variables)
    actor_optimizer.apply_gradients(
        zip(actor_grad, policy.model.trainable_variables)
    )
  

#######################################################################################################33
#######################################################################################################33
#######################################################################################################33

def rendertrial(maxiter=NSTEPS,verbose=True):
    '''
    Display a roll-out from random start and optimal feedback.
    Press ^Z to get a roll-out at training time.
    '''
    x = env.reset()
    rsum = 0.
    for i in range(maxiter):
        u = policy(x)
        x, reward = env.step(u)[:2]
        env.render()
        rsum += reward
    if verbose: print('Lasted ',i,' timestep -- total reward:',rsum)
signal.signal(signal.SIGTSTP, lambda x,y:rendertrial()) # Roll-out when CTRL-Z is pressed
env.full.sleepAtDisplay=5e-3

# Logs
h_rewards = []
h_steps   = []

# Takes about 4 min to train
for episode in range(NEPISODES):

    prev_state = env.reset()

    for step in range(NSTEPS):
    # Uncomment this to see the Actor in action
        # But not in a python notebook.
        #env.render()

        action = policy(prev_state, noise=ou_noise())
        state, reward = env.step(action)[:2]
        done=False
        
        replayDeque.append(ReplayItem(prev_state, action, reward, done, state))
        
        prev_state = state

        if len(replayDeque) <= BATCH_SIZE:  continue


        ####################################################################
        # Sample a minibatch
        
        batch = random.sample(replayDeque,BATCH_SIZE)            # Random batch from replay memory.
        state_batch    = tf.convert_to_tensor([ b.x      for b in batch ])
        action_batch    = tf.convert_to_tensor([ b.u      for b in batch ])
        reward_batch    = tf.convert_to_tensor([ [ b.reward ] for b in batch ],dtype=np.float32)
        done_batch    = tf.convert_to_tensor([ b.done   for b in batch ])
        next_state_batch   = tf.convert_to_tensor([ b.x2     for b in batch ])

        ####################################################################
        # One gradient step for the minibatch

        # Critic and actor gradients
        learn(state_batch, action_batch, reward_batch, next_state_batch)
        # Step smoothing using target networks
        policy.targetAssign(policyTarget)
        quality.targetAssign(qualityTarget)

        if done: break   # stop at episode end.

    # Some prints and logs
    episodic_reward = sum([ replayDeque[-i-1].reward for i in range(step+1) ])
    h_rewards.append( episodic_reward )
    h_steps.append(step+1)
    
    print(f'Ep#{episode:3d}: lasted {step+1:d} steps, reward={episodic_reward:3.1f} ')

    
    # avg_reward = np.mean(h_rewards[-40:])
    # if episode==5 and RANDOM_SEED==0:
    #     assert(  abs(avg_reward + 1423.0528188196286) < 1e-3 )
    # if episode==0 and RANDOM_SEED==0:
    #     assert(  abs(avg_reward + 1712.386325099637) < 1e-3 )
        
# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(h_rewards)
plt.xlabel("Episode")
plt.ylabel("Epsiodic Reward")
plt.show()

#######################################################################################################33
#######################################################################################################33
#######################################################################################################33
