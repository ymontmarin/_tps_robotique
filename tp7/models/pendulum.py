'''
Create a simulation environment for a N-pendulum.
Example of use:

env = Pendulum(N)
env.reset()

for i in range(1000):
   env.step(zero(env.nu))
   env.render()

'''

from pinocchio.utils import *
import pinocchio as pin
import hppfcl

# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------
# --- PENDULUM ND CONTINUOUS --------------------------------------------------------------------

def Capsule(name,joint,radius,length,placement,color=[.7,.7,0.98,1]):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    hppgeom = hppfcl.Capsule(radius,length)
    geom = pin.GeometryObject(name,joint,hppgeom,placement)
    geom.meshColor = np.array(color)
    return geom


def createPendulum(nbJoint,length=1.0,mass=1.0,viewer=None):
    '''
    Creates the Pinocchio kinematic <rmodel> and visuals <gmodel> models for 
    a N-pendulum.

    @param nbJoint: number of joints <N> of the N-pendulum.
    @param length: length of each arm of the pendulum.
    @param mass: mass of each arm of the pendulum.
    @param viewer: gepetto-viewer CORBA client. If not None, then creates the geometries
    in the viewer.
    '''
    rmodel = pin.Model()
    gmodel = pin.GeometryModel()
    
    color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
    colorred = [1.0,0.0,0.0,1.0]

    radius = 0.1*length
    
    prefix = ''
    jointId = 0
    jointPlacement = pin.SE3.Identity()
    inertia = pin.Inertia(mass,
                          np.matrix([0.0,0.0,length/2]).T,
                          mass/5*np.diagflat([ length**2, 1e-2, 1e-2 ]) )

    for i in range(nbJoint):
        # Add a new joint
        istr = str(i)
        name               = prefix+"joint"+istr
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointId = rmodel.addJoint(jointId,pin.JointModelRX(),jointPlacement,jointName)
        rmodel.appendBodyToJoint(jointId,inertia,pin.SE3.Identity())
        jointPlacement     = pin.SE3(eye(3),np.matrix([0.0,0.0,length]).T)

        # Add a red sphere for visualizing the joint.
        gmodel.addGeometryObject(Capsule(jointName,jointId,1.5*radius,0.0,pin.SE3.Identity(),colorred))
        # Add a white segment for visualizing the link.
        gmodel.addGeometryObject(Capsule(bodyName ,jointId,radius,0.8*length,
                                         pin.SE3(eye(3),np.matrix([0.,0.,length/2]).T),
                                         color))

    rmodel.addFrame( pin.Frame('tip',jointId,0,jointPlacement,pin.FrameType.OP_FRAME) )

    rmodel.upperPositionLimit = np.zeros(nbJoint)+2*np.pi
    rmodel.lowerPositionLimit = np.zeros(nbJoint)-2*np.pi
    rmodel.velocityLimit      = np.zeros(nbJoint)+5.0
    
    return rmodel,gmodel

def createPendulumWrapper(nbJoint,initViewer=True):
    '''
    Returns a RobotWrapper with a N-pendulum inside.
    '''
    rmodel,gmodel = createPendulum(nbJoint)
    rw = pin.RobotWrapper(rmodel,visual_model=gmodel,collision_model=gmodel)
    return rw

if __name__ == "__main__":
    rw = createPendulumWrapper(3,True)
    rw.initViewer(loadModel=True)
    rw.display(np.random.rand(3)*6-3)

