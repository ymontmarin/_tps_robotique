'''
Base (abstract) class for environments.
'''
import numpy as np
import time
from tp7.discretization import VectorDiscretization

def scalarOrArrayToArray(x,nx):
    return x if isinstance(x,np.ndarray) else np.array([x,]*nx,np.float64)

# ----------------------------------------------------------------------------------------
# --- Env Abstract -----------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class EnvAbstract:
    '''
    Base (abstract) class for environments.
    '''
    def __init__(self,nx,nu):
        '''
        '''
        self.nx = nx
        self.nu = nu

    def render(self):
        '''
        Call self.display(self.x).
        '''
        self.display(self.x)
        
    def reset(self,x=None):
        '''
        This method internally calls self.randomState() and stores the results. 
        '''
        if x is None: self.x = self.randomState()
        else:         self.x = x.copy() if isinstance(x,np.ndarray) else x
        return self.x
        
    def step(self,u):
        '''
        This methods calls self.x,self.cost = self.dynAndCost(self.x,u).
        Modifies the internal state self.x.
        '''
        self.x,cost = self.dynAndCost(self.x,u)
        return self.x,-cost
    
    # Internal methods corresponding to reset (randomState), step (cost and dyn) and
    # render (display). They are all to be used with continuous state and control, and are
    # read only (no change of internal state).
    def randomState(self):
        '''
        Returns a random state x. 
        Const method, dont modify the environment (i.e env.x is not modified).
        '''
        assert(False and "This method should be implemented by inheritance.")

    def dynAndCost(self,x,u):
        '''
        Considering a control u applied at current state x, the method returns
        (xnext,cost), where xnext is the next state xnext = f(x,u), and cost
        is the cost for making this action cost=l(x,u).
        This method is called inside step(u).
        Const method, dont modify the environment (i.e env.x is not modified).
        '''
        assert(False and "This method should be implemented by inheritance.")

    def display(self,x):
        '''
        Display the argument state x (not the internal env.x).
        This method is called inside env.render().
        Const method, dont modify the environment (i.e env.x is not modified).
        '''
        assert(False and "This method should be implemented by inheritance.")
        
       
# ----------------------------------------------------------------------------------------
# --- CONTINUOUS ENV ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class EnvContinuousAbstract(EnvAbstract):
    def __init__(self,nx,nu,xmax=10.,xmin=None,umax=1.,umin=None):
        EnvAbstract.__init__(self,nx,nu)
        self.xmax = scalarOrArrayToArray(xmax,self.nx)
        self.xmin = scalarOrArrayToArray(xmin,self.nx) if xmin is not None else -self.xmax
        self.umax = scalarOrArrayToArray(umax,self.nu)
        self.umin = scalarOrArrayToArray(umin,self.nu) if umin is not None else -self.umax
        self.xspan = self.xmax-self.xmin
        self.uspan = self.umax-self.umin

    def randomState(self):
        return self.xmin+np.random.random(self.nx)*self.xspan
    def randomControl(self):
        return self.umin+np.random.random(self.nu)*self.uspan

import pinocchio as pin
class EnvPinocchio(EnvContinuousAbstract):
    def __init__(self,pinocchioModel,viewer=None,taumax=1.0):
        self.rmodel = pinocchioModel
        self.rdata  = self.rmodel.createData()
        self.nq = self.rmodel.nq
        self.nv = self.rmodel.nv
        qmax = self.rmodel.upperPositionLimit
        qmin = self.rmodel.lowerPositionLimit
        vmax = self.rmodel.velocityLimit
        vmin = -vmax
        self.viewer = viewer
        EnvContinuousAbstract.__init__(self,nx=self.nq+self.nv,nu=self.nv,
                                       xmax=np.concatenate([qmax.ravel(),vmax.ravel()]),
                                       xmin=np.concatenate([qmin.ravel(),vmin.ravel()]),
                                       umax=taumax)
        # Options parameters
        self.sleepAtDisplay = 8e-2
        self.xdes = np.zeros(self.nx)
        self.costWeights = { 'wx': 1., 'wu': 1e-2 } 
        self.DT  = 5e-2   # Step length
        self.NDT = 2      # Number of Euler steps per integration (internal)
        self.Kf  = .1     # Friction coefficient (remove it with Kf=0).
        self.reset()

    def randomState(self):
        #q = pin.randomConfiguration(self.rmodel)
        dq = self.xmax[:self.nq]-self.xmin[:self.nq]
        q = np.random.random(self.nq)*dq + self.xmin[:self.nq]
        dv = self.xmax[-self.nv:]-self.xmin[-self.nv:]
        v = np.random.random(self.nv)*dv+self.xmin[-self.nv:]
        return np.concatenate([q,v])
    def display(self,x,sleep=None):
        if sleep is not None: self.sleepAtDisplay = sleep
        q,v = x[:self.nq],x[-self.nv:]
        if self.viewer is not None:
            self.viewer.display(q)
            time.sleep(self.sleepAtDisplay)
    def cost(self,x,u):
        '''Default cost function.'''
        cost  = 0.
        cost += self.costWeights['wx']*np.sum((x-self.xdes)**2)
        cost += self.costWeights['wu']*np.sum(u**2)
        return cost
    def dynAndCost(self,x,u,verbose=False):
        x=x.copy()
        q,v = x[:self.nq],x[-self.nv:]
        u = np.clip(u,-self.umax,self.umax)
        cost = 0.
        dt = self.DT/self.NDT
        for t in range(self.NDT):
            # Add friction
            tau = u.copy()
            if self.Kf>0.0: tau -= self.Kf*v
            # Evaluate cost
            cost  += self.cost(np.concatenate([q,v]),u)/self.NDT
            #print(self,self.cost,t,x,u,cost)
            # Evaluate dynamics
            a = pin.aba(self.rmodel,self.rdata,q,v,tau)
            if verbose: print(q,v,tau,a)
            v += a*dt
            v = np.clip(v,self.xmin[self.nq:],self.xmax[self.nq:])
            q = pin.integrate(self.rmodel,q,v*dt)
        xnext = np.concatenate([q,v])
        return xnext,cost

# ----------------------------------------------------------------------------------------
# --- PARTIALLY OBSERVABLE ---------------------------------------------------------------
# ----------------------------------------------------------------------------------------
class EnvPartiallyObservable(EnvContinuousAbstract):
    def __init__(self,env_fully_observable,nobs,obs,obsInv=None):
        '''
        Define a partially-observable markov model from a fully-observable model
        and an observation function.
        The new env model is defined with the observable as new state, while
        the original state is kept inside the fully-observable model.

        @param env_fully_observable: the fully-observable model.
        @param obs: the observation function y=h(x)
        @param obsinv: if available, the inverse function of h: x=h^-1(y).
        '''
        self.full = env_fully_observable
        self.obs = obs
        self.obsinv = obsInv
        EnvContinuousAbstract.__init__(self,nx=nobs,
                                       nu=self.full.nu,
                                       xmax=obs(self.full.xmax),
                                       xmin=obs(self.full.xmin),
                                       umax=self.full.umax,umin=self.full.umin)

    def randomState(self):
        return self.obs(self.full.randomState())
    def dynAndCost(self,x,u):
        assert(self.obsinv is not None)
        x,c = self.full.dynAndCost(self.obsinv(x),u)
        return self.obs(x),c
    def display(self,x):
        assert(self.obsinv is not None)
        self.full.display(self.obsinv(x))

    def reset(self,x=None):
        assert(x is not None or self.obsinv is not None)
        return self.obs(self.full.reset(x))
    def step(self,u):
        x,c = self.full.step(u)
        return self.obs(x),c
    def render(self):
        return self.full.render()
    @property
    def x(self):
        return self.obs(self.full.x)
        


# ----------------------------------------------------------------------------------------
# --- DISCRETIZED ENV --------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

class EnvDiscretized(EnvAbstract):
    def __init__(self,envContinuous,discretize_x = 0,discretize_u = 0):
        self.conti = envContinuous
        if discretize_u != 0:
            self.discretize_u = VectorDiscretization(self.conti.nu,
                                                     vmax=self.conti.umax,nsteps=discretize_u)
            self.encode_u = self.discretize_u.c2i
            self.decode_u = self.discretize_u.i2c
            nu = self.discretize_u.nd
        else:
            self.discretize_u = None
            self.encode_u = lambda u:u
            self.decode_u = lambda u:u
            nu = envContinuous.nu
        if discretize_x != 0:
            self.discretize_x = VectorDiscretization(self.conti.nx,
                                                     vmax=self.conti.xmax,nsteps=discretize_x)
            self.encode_x = self.discretize_x.c2i
            self.decode_x = self.discretize_x.i2c
            nx = self.discretize_x.nd
        else:
            self.discretize_x = None
            self.encode_x = lambda x:x
            self.decode_x = lambda x:x
            nx = envContinuous.nx

        EnvAbstract.__init__(self,nx=nx,nu=nu)

    def randomState(self):
        return self.encode_x(self.conti.randomState())
    def display(self,x):
        self.conti.display(self.decode_x(x))
    def dynAndCost(self,x,u):
        x,c=self.conti.dynAndCost(self.decode_x(x),self.decode_u(u))
        return self.encode_x(x),c
    def reset(self,xi=None):
        if xi is None:
            x = self.conti.reset()
            xi = self.encode_x(x)
        else:
            x = None
        x_eps = self.decode_x(xi)
        if x_eps is not x:
            self.conti.reset(x_eps)
        self.x = xi
        return self.x
    def step(self,u):
        ###assert(self.x == self.encode_x(self.conti.x))
        ###assert(np.allclose(self.decode_x(self.x),self.conti.x))
        x,c   = self.conti.step(self.decode_u(u))
        self.x= self.encode_x(x)
        x_eps = self.decode_x(self.x)
        if x_eps is not x: self.conti.reset(x_eps)
        return self.x,c
    def render(self):
        self.conti.render()

        

                
