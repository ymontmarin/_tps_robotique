# %jupyter_snippet import
import pinocchio as pin
from utils.meshcat_viewer_wrapper import MeshcatVisualizer
import time
import numpy as np
from numpy.linalg import inv,norm,pinv,svd,eig
from scipy.optimize import fmin_bfgs,fmin_slsqp
from utils.load_ur5_with_obstacles import load_ur5_with_obstacles,Target
import matplotlib.pylab as plt
# %end_jupyter_snippet
plt.ion()  # matplotlib with interactive setting

# %jupyter_snippet robot
robot = load_ur5_with_obstacles(reduced=True)
# %end_jupyter_snippet

# %jupyter_snippet viewer
viz = MeshcatVisualizer(robot)
viz.display(robot.q0)
# %end_jupyter_snippet

# %jupyter_snippet target
target = Target(viz,position = np.array([.5,.5]))
# %end_jupyter_snippet

################################################################################
################################################################################
################################################################################

# %jupyter_snippet endef
def endef(q):
     '''Return the 2d position of the end effector.'''
     pin.framesForwardKinematics(robot.model,robot.data,q)
     return robot.data.oMf[-1].translation[[0,2]]
# %end_jupyter_snippet

# %jupyter_snippet  dist
def dist(q):
     '''Return the distance between the end effector end the target (2d).'''
     return norm(endef(q)-target.position)
# %end_jupyter_snippet

# %jupyter_snippet coll
def coll(q):
     '''Return true if in collision, false otherwise.'''
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     return pin.computeCollisions(robot.collision_model,robot.collision_data,False)
# %end_jupyter_snippet

# %jupyter_snippet qrand
def qrand(check=False):
    '''
    Return a random configuration. If check is True, this
    configuration is not is collision
    '''
    while True:
        q = np.random.rand(2)*6.4-3.2  # sample between -3.2 and +3.2.
        if not check or not coll(q):
            return q
# %end_jupyter_snippet

# %jupyter_snippet colldist
def collisionDistance(q):
     '''Return the minimal distance between robot and environment. '''
     pin.updateGeometryPlacements(robot.model,robot.data,robot.collision_model,robot.collision_data,q)
     if pin.computeCollisions(robot.collision_model,robot.collision_data,False): 0
     idx = pin.computeDistances(robot.collision_model,robot.collision_data)
     return robot.collision_data.distanceResults[idx].min_distance
# %end_jupyter_snippet
################################################################################
################################################################################
################################################################################

# %jupyter_snippet qrand_target
# Sample a random free configuration where dist is small enough.
def qrandTarget(threshold=5e-2, display=False):
     while True:
          q = qrand()
          if display:
               viz.display(q)
               time.sleep(1e-3)
          if not coll(q) and dist(q)<threshold:
               return q
viz.display(qrandTarget())
# %end_jupyter_snippet

################################################################################
################################################################################
################################################################################

# %jupyter_snippet random_descent
# Random descent: crawling from one free configuration to the target with random
# steps.
def randomDescent(q0 = None):
     q = qrand(check=True) if q0 is None else q0
     hist = [ q.copy() ]
     for i in range(100):
          dq = qrand()*.1                           # Choose a random step ...
          qtry = q+dq                               # ... apply
          if dist(q)>dist(q+dq) and not coll(q+dq): # If distance decrease without collision ...
               q = q+dq                             # ... keep the step
               hist.append(q.copy())                # ... keep a trace of it
               viz.display(q)                       # ... display it
               time.sleep(5e-3)                     # ... and sleep for a short while
     return hist
# %end_jupyter_snippet

################################################################################
################################################################################
################################################################################

# %jupyter_snippet sample
def sampleSpace(nbSamples=500):
     '''
     Sample nbSamples configurations and store them in two lists depending
     if the configuration is in free space (hfree) or in collision (hcol), along
     with the distance to the target and the distance to the obstacles.
     '''
     hcol = []
     hfree = []
     for i in range(nbSamples):
          q = qrand(False)
          if not coll(q):
               hfree.append( list(q.flat) + [ dist(q), collisionDistance(q) ])
          else:
               hcol.append(  list(q.flat) + [ dist(q), 1e-2 ])
     return hcol,hfree

def plotConfigurationSpace(hcol,hfree,markerSize=20):
     '''
     Plot 2 "scatter" plots: the first one plot the distance to the target for 
     each configuration, the second plots the distance to the obstacles (axis q1,q2, 
     distance in the color space).
     '''
     htotal = hcol + hfree
     h=np.array(htotal)
     plt.subplot(2,1,1)
     plt.scatter(h[:,0],h[:,1],c=h[:,2],s=markerSize,lw=0)
     plt.title("Distance to the target")
     plt.colorbar()
     plt.subplot(2,1,2)
     plt.scatter(h[:,0],h[:,1],c=h[:,3],s=markerSize,lw=0)
     plt.title("Distance to the obstacles")
     plt.colorbar()

hcol,hfree = sampleSpace(100)
plotConfigurationSpace(hcol,hfree)
# %end_jupyter_snippet

################################################################################
################################################################################
################################################################################

### Plot random trajectories in the same plot
# %jupyter_snippet traj
qinit = np.array([-1.1, -3. ])
for i in range(100):
     traj = randomDescent(qinit)
     if dist(traj[-1])<5e-2:
          print('We found a good traj!')
          break
traj = np.array(traj)
### Chose trajectory end to be in [-pi,pi]
qend = (traj[-1]+np.pi) % (2*np.pi) - np.pi
### Take the entire trajectory it modulo 2 pi
traj += (qend-traj[-1])
# %end_jupyter_snippet
plt.plot(traj[:,0],traj[:,1],'r',lw=5)

################################################################################
################################################################################
################################################################################

# %jupyter_snippet optim
def cost(q):
    """
    Cost function: distance to the target.
    """
    return dist(q) ** 2


def constraint(q):
    """
    Constraint function: distance to the obstacle should be strictly positive.
    """
    min_collision_dist = 0.01  # [m]
    return collisionDistance(q) - min_collision_dist


def callback(q):
    """
    At each optimization step, display the robot configuration.
    """
    viz.display(q)
    time.sleep(0.01)


def optimize():
     '''
     Optimize from an initial random configuration to discover a collision-free
     configuration as close as possible to the target. 
     '''
     return fmin_slsqp(x0=qrand(check=True),
                       func=cost,
                       f_ieqcons=constraint,callback=callback,
                       full_output=1)
optimize()
# %end_jupyter_snippet

# %jupyter_snippet useit
while True:
    res=optimize()
    q=res[0]
    viz.display(q)
    if res[4]=='Optimization terminated successfully' and res[1]<1e-6:
        print('Finally successful!')
        break
    print("Failed ... let's try again! ")
# %end_jupyter_snippet
