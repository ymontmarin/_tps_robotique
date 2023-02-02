'''
Create a simulation environment for a N-pendulum.
See the main at the end of the file for an example.

We define here 4 main environments that are tested in the __main__:

- EnvPendulum:    state NX=2 continuous, control NU=1 continuous, Euler integration step with DT=1e-2 and high friction
- EnvPendulumDiscrete:  state NX=441 discrete, control NU=11 discrete, Euler step DT=0.5 low friction
- EnvPendulumSinCos: state NX=3 with x=[cos,sin,vel], control NU=1 control, Euler step DT=1e-2, high friction
- EnvPendulumHybrid:  state NX=3 continuous with x=[cos,sin,vel], control NU=11 discrete, Euler step DT=0.5 low friction

'''

import pinocchio as pin
import numpy as np
from tp7.models.pendulum import createPendulum
from tp7.env_abstract import EnvPinocchio
import tp7.env_abstract as env_abstract

# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
        
class EnvPendulum(EnvPinocchio):
    '''
    Define a class Robot with 7DOF (shoulder=3 + elbow=1 + wrist=3). 
    The configuration is nq=7. The velocity is the same. 
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).
    
    See tp1.py for an example of use.
    '''

    def __init__(self,nbJoint=1,viewer=None):
        '''
        Create a Pinocchio model of a N-pendulum, with N the argument <nbJoint>.
        <viewer> is expected to be in { "meshcat" | "gepetto" | None }.
        '''
        self.model,self.geometryModel = createPendulum(nbJoint)
        if viewer == 'meshcat':
            from pinocchio.visualize import MeshcatVisualizer
            self.viewer = MeshcatVisualizer(self.model,self.geometryModel,self.geometryModel)
            self.viewer.initViewer(loadModel=True)
        elif 'gepetto' == viewer:
            from pinocchio.visualize import GepettoVisualizer
            self.viewer = GepettoVisualizer(self.model,self.geometryModel,self.geometryModel)
            self.viewer.initViewer(loadModel=True)
        else:
            self.viewer = None
            if viewer is not None:  ## warning
                print('Error:  <viewer> is expected to be in { "meshcat" | "gepetto" | None }, but we got:', viewer)
                    
        self.q0         = pin.neutral(self.model)
        self.v0         = np.zeros(self.model.nv)
        self.x0         = np.concatenate([self.q0,self.v0])

        EnvPinocchio.__init__(self,self.model,self.viewer,taumax=2.5)
        self.DT = 1e-2    # duration of one environment step
        self.NDT = 5      # number of euler integration step per environment step (i.e integration interval is DT/NDT)
        self.Kf = 1.0     # Friction coefficient

        self.costWeights = { 'q': 1, 'v' : 1e-1, 'u' : 1e-3, 'tip' : 0. }
        self.tipDes = float(nbJoint)
        
    def cost(self,x=None,u=None):
        if x is None: x = self.x
        cost  = 0.
        q,v = x[:self.nq],x[-self.nv:]
        qdes = self.xdes[:self.nq]
        cost += self.costWeights['q']*np.sum((q-qdes)**2)
        cost += self.costWeights['v']*np.sum(v**2)
        cost += 0 if u is None else self.costWeights['u']*np.sum(u**2)
        cost += self.costWeights['tip']*(self.tip(q)-self.tipDes)**2
        return cost

    def tip(self,q=None):
        '''Return the altitude of pendulum tip'''
        if q is None: q = self.x[:self.nq]
        pin.framesForwardKinematics(self.rmodel,self.rdata,q)
        return self.rdata.oMf[1].translation[2]

    def jupyter_cell(self):
        return self.viewer.viewer.jupyter_cell()
    
# --- SPIN-OFF ----------------------------------------------------------------------------------
# --- SPIN-OFF ----------------------------------------------------------------------------------
# --- SPIN-OFF ----------------------------------------------------------------------------------

class EnvPendulumDiscrete(env_abstract.EnvDiscretized):
    def __init__(self,nbJoint=1,**kwargs):
        env = EnvPendulum(nbJoint,**kwargs)
        env.DT=5e-1     # Larger integration step to allow larger discretization grid
        env.Kf=0.1      # Reduced friction, because larger steps would make friction unstable.
        env_abstract.EnvDiscretized.__init__(self,env,21,11)
        self.discretize_x.modulo = np.pi*2
        self.discretize_x.moduloIdx = range(env.nq)
        self.discretize_x.vmax[:env.nq] = np.pi
        self.discretize_x.vmin[:env.nq] = -np.pi
        self.reset()
        self.conti.costWeights = { 'q': 0, 'v' : 0, 'u' : 0, 'tip' : 1 }
        self.withSimpleCost = True
    def step(self,u):
        x,c=env_abstract.EnvDiscretized.step(self,u)
        if self.withSimpleCost:
            c = int(np.all(np.abs(self.conti.x)<1e-3))
        return x,c
    def jupyter_cell(self):
        return self.conti.viewer.viewer.jupyter_cell()
        
class EnvPendulumSinCos(env_abstract.EnvPartiallyObservable):
    def __init__(self,nbJoint=1,**kwargs):
        env = EnvPendulum(nbJoint,**kwargs)
        def sincos(x,nq):
            q,v = x[:nq],x[nq:]
            return np.concatenate([np.concatenate([(np.cos(qi),np.sin(qi)) for qi in q]),v])
        def atan(x,nq):
            cq,sq,v = x[:2*nq:2],x[1:2*nq:2],x[2*nq:]
            return np.concatenate([np.arctan2(sq,cq),v])
        env_abstract.EnvPartiallyObservable.__init__(self,env,nobs=env.nq*2+env.nv,
                                                     obs=lambda x:sincos(x,env.nq),
                                                     obsInv=lambda csv:atan(csv,env.nq))
        self.reset()
    def jupyter_cell(self):
        return self.full.viewer.jupyter_cell()
        
class EnvPendulumHybrid(env_abstract.EnvDiscretized):
    def __init__(self,nbJoint=1,**kwargs):
        env = EnvPendulumSinCos(nbJoint,**kwargs)
        NU = 21 #11
        env_abstract.EnvDiscretized.__init__(self,env,discretize_x=0,discretize_u=NU)
        self.conti.full.Kf=.1   # Reduced friction, because larger steps would make friction unstable.
        self.conti.full.DT=1e-1 # 5e-1 # Larger integration step to allow larger discretization grid
        self.conti.full.costWeights = {'q': 0, 'tip': 1.0, 'u': 0.00, 'v': 0.1 }

        self.reset()
    #     self.withSimpleCost = False
    # def step(self,u):
    #     x,c=env_abstract.EnvDiscretizedenv_abstract.step(self,u)
    #     if self.withSimpleCost:
    #         c = int(np.all(np.abs(self.conti.x)<1e-3))
    #     return x,c
    def jupyter_cell(self):
        return self.conti.full.viewer.viewer.jupyter_cell()

        
# --- MAIN -------------------------------------------------------------------------------
# --- MAIN -------------------------------------------------------------------------------
# --- MAIN -------------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    envs = []
    
    env = EnvPendulum(1,viewer='gepetto')
    env.name = str(env.__class__)
    env.u0 = np.zeros(env.nu)
    envs.append(env)
    
    env = EnvPendulumDiscrete(1,viewer='gepetto')
    env.name = str(env.__class__)
    env.u0 = env.encode_u(np.zeros(1))
    envs.append(env)

    env = EnvPendulumSinCos(1,viewer='gepetto')
    env.name = str(env.__class__)
    env.u0 = np.zeros(env.nu)
    envs.append(env)

    env = EnvPendulumHybrid(1,viewer='gepetto')
    env.name = str(env.__class__)
    env.u0 = env.encode_u(np.zeros(1))
    envs.append(env)

    # Reset all environment to the same initial state.
    envs[1].reset()
    for env in envs:
        if env is not envs[1]: env.reset(envs[1].conti.x)

    # Simulate a free fall for all environments.
    for env in envs:
        env.render()
        print(env.name)
        time.sleep(1)
        for i in range(10):
            env.step(env.u0)
            env.render()
