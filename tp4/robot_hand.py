from pinocchio.explog import exp,log
from numpy.linalg import pinv,norm,pinv
from pinocchio.utils import rotate
import pinocchio as pin
import numpy as np
from numpy import pi
from numpy import cos,sin,pi,hstack,vstack,argmin
import hppfcl

def Capsule(name,joint,radius,length,placement,color=[.7,.7,0.98,1]):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    ### They should be capsules ... but hppfcl current version is buggy with Capsules...
    #hppgeom = hppfcl.Capsule(radius,length)
    hppgeom = hppfcl.Cylinder(radius,length)
    geom = pin.GeometryObject(name,joint,hppgeom,placement)
    geom.meshColor = np.array(color)
    return geom

def Sphere(name,joint,radius,placement,color=[.9,.9,0.98,1]):
    '''Create a Pinocchio::FCL::Capsule to be added in the Geom-Model. '''
    hppgeom = hppfcl.Sphere(radius)
    geom = pin.GeometryObject(name,joint,hppgeom,placement)
    geom.meshColor = np.array(color)
    return geom



class RobotHand:
    '''
    Define a class Robot with 7DOF (shoulder=3 + elbow=1 + wrist=3). 
    The configuration is nq=7. The velocity is the same. 
    The members of the class are:
    * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
    * model: the kinematic tree of the robot.
    * data: the temporary variables to be used by the kinematic algorithms.
    * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
    an object Visual (see above).
    
    CollisionPairs is a list of visual indexes. 
    Reference to the collision pair is used in the collision test and jacobian of the collision
    (which are simply proxy method to methods of the visual class).
    '''

    def __init__(self):
        self.viewer = None
        self.model = pin.Model()
        self.gmodel = pin.GeometryModel()

        self.createHand()
        self.addCollisionPairs()

        self.data = self.model.createData()
        self.gdata = pin.GeometryData(self.gmodel)
        self.gdata.collisionRequests.enable_contact=True
        
        self.q0 = np.zeros(self.model.nq)
        self.q0[-2] = np.pi/3
        self.q0[2:-4] = np.pi/6
        self.q0[11:] = np.pi/4
        self.v0 = np.zeros(self.model.nv)
        self.collisionPairs = []


        self.collision_model = self.gmodel
        self.collision_data = self.gmodel.createData()
        self.visual_model = self.gmodel
        self.visual_data = self.gmodel.createData()
        
    def addCollisionPairs(self):
        
        pairs = [

            [ 'finger12', 'wrist' ],
            [ 'finger12', 'palm_left' ],
            [ 'finger12', 'palm_right' ],
            [ 'finger12', 'palm_front' ],

            [ 'finger13', 'wrist' ],
            [ 'finger13', 'palm_left' ],
            [ 'finger13', 'palm_right' ],
            [ 'finger13', 'palm_front' ],
            [ 'finger13', 'palm2' ],

            [ 'finger22', 'wrist' ],
            [ 'finger22', 'palm_left' ],
            [ 'finger22', 'palm_right' ],
            [ 'finger22', 'palm_front' ],

            [ 'finger23', 'wrist' ],
            [ 'finger23', 'palm_left' ],
            [ 'finger23', 'palm_right' ],
            [ 'finger23', 'palm_front' ],
            [ 'finger23', 'palm2' ],

            [ 'finger32', 'wrist' ],
            [ 'finger32', 'palm_left' ],
            [ 'finger32', 'palm_right' ],
            [ 'finger32', 'palm_front' ],

            [ 'finger33', 'wrist' ],
            [ 'finger33', 'palm_left' ],
            [ 'finger33', 'palm_right' ],
            [ 'finger33', 'palm_front' ],
            [ 'finger33', 'palm2' ],

            [ 'thumb1', 'wrist' ],
            [ 'thumb1', 'palm_left' ],
            [ 'thumb1', 'palm_front' ],
            [ 'thumb1','palm2' ],
            [ 'thumb1','finger11'],
            [ 'thumb1','finger12'],
            [ 'thumb1','finger13'],

            [ 'thumb2', 'wrist' ],
            [ 'thumb2', 'palm_left' ],
            [ 'thumb2', 'palm_right' ],
            [ 'thumb2', 'palm_front' ],
            [ 'thumb2', 'palm2' ],
            [ 'thumb2','finger11'],
            [ 'thumb2','finger12'],
            [ 'thumb2','finger13'],
            [ 'thumb2','finger21'],
            [ 'thumb2','finger22'],
            [ 'thumb2','finger23'],
        ]
        for (n1,n2) in pairs:
            self.gmodel.addCollisionPair(pin.CollisionPair(self.gmodel.getGeometryId('world/'+n1),
                                                           self.gmodel.getGeometryId('world/'+n2)))
        
    def createHand(self,rootId=0,jointPlacement=None):
        color   = [red,green,blue,transparency] = [1,1,0.78,1.0]
        colorred = [1.0,0.0,0.0,1.0]

        jointId = rootId

        cm = 1e-2
        trans = lambda x,y,z: pin.SE3(np.eye(3),np.array([x,y,z]))
        inertia = lambda m,c: pin.Inertia(m,np.array(c,np.double),np.eye(3)*m**2)

        name               = "wrist"
        jointName,bodyName = [name+"_joint",name+"_body"]
        #jointPlacement     = jointPlacement if jointPlacement!=None else pin.SE3.Identity()
        jointPlacement     = jointPlacement if jointPlacement!=None else pin.SE3(pin.utils.rotate('y',np.pi),np.zeros(3))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(3,[0,0,0]),pin.SE3.Identity())

        ## Hand dimsensions: length, width, height(=depth), finger-length
        L=3*cm;W=5*cm;H=1*cm; FL = 4*cm
        self.gmodel.addGeometryObject( Sphere('world/wrist',jointId,.02,
                                              pin.SE3(rotate('x',pi/2),np.array([0,0,0]))) )
        
        self.gmodel.addGeometryObject( Capsule('world/palm_right',jointId,H,L,
                                               pin.SE3(rotate('z',-.3)@rotate('y',pi/2),np.array([L/2,-W/2.6,0])) ) )
        self.gmodel.addGeometryObject( Capsule('world/palm_left',jointId,H,L,
                                               pin.SE3(rotate('y',pi/2),np.array([L/2,W/2,0]))) )
        self.gmodel.addGeometryObject( Capsule('world/palm_front',jointId,H,W,
                                               pin.SE3(rotate('x',pi/2),np.array([L,0,0]))) )
        self.gmodel.addGeometryObject( Sphere('world/palm_frontleft',jointId,H,
                                              pin.SE3(rotate('x',pi/2),np.array([L,W/2,0])),
                                              color=[.7,.7,0.98,1]) )
        self.gmodel.addGeometryObject( Sphere('world/palm_frontright',jointId,H,
                                              pin.SE3(rotate('x',pi/2),np.array([L,-W/2,0])),
                                              color=[.7,.7,0.98,1]) )
        
        name               = "palm"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([5*cm,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(2,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/palm2',jointId,1*cm,W,
                                               pin.SE3(rotate('x',pi/2),np.zeros(3))) )
        palmIdx = jointId

        name               = "finger11"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([2*cm,W/2,0]))
        jointId = self.model.addJoint(palmIdx,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger11',jointId,H,FL-0*H,
                                        pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0])) ))

        name               = "finger12"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger12',jointId,H,FL-0*H,
                                    pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0])) ))


        name               = "finger13"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL-H,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Sphere('world/finger13',jointId,H,
                                       trans(H,0,0) ))

        name               = "finger21"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([2*cm,0,0]))
        jointId = self.model.addJoint(palmIdx,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger21',jointId,H,FL-0*H,
                                        pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0])) ))

        name               = "finger22"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger22',jointId,H,FL-0*H,
                                    pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0])) ))

        name               = "finger23"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL-H,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Sphere('world/finger23',jointId,H,
                                       trans(H,0,0) ))

        name               = "finger31"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([2*cm,-W/2,0]))
        jointId = self.model.addJoint(palmIdx,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger31',jointId,H,FL-0*H,
                                        pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0]))))

        name               = "finger32"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/finger32',jointId,H,FL-0*H,
                                        pin.SE3(rotate('y',pi/2),np.array([FL/2-H,0,0]))))

        name               = "finger33"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.array([FL-H,0,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.3,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Sphere('world/finger33',jointId,H,
                                        trans(H,0,0)))

        name               = "thumb1"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(rotate('z',-1.5), np.array([1*cm,-W/2-H*1.3,0]))
        jointId = self.model.addJoint(1,pin.JointModelRY(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        # self.gmodel.addGeometryObject( Capsule('world/thumb1_mid',jointId,H,2*cm,
        #                                        pin.SE3(rotate('z',pi/3)@rotate('x',pi/2),np.array([1*cm,-1*cm,0])) ))
        
        name               = "thumb1a"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(np.eye(3), np.zeros(3))
        jointId = self.model.addJoint(jointId,pin.JointModelRX(),jointPlacement,jointName)
        # self.model.appendBodyToJoint(jointId,inertia(.5,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/thumb1',jointId,H,2*cm,
                                               pin.SE3(rotate('z',pi/3)@rotate('x',pi/2),np.array([0.3*cm,-1.0*cm,0]))))

        name               = "thumb2"
        jointName,bodyName = [name+"_joint",name+"_body"]
        jointPlacement     = pin.SE3(rotate('z',pi/3)@rotate('x',pi), np.array([3.4*cm,-1.99*cm,0]))
        jointId = self.model.addJoint(jointId,pin.JointModelRZ(),jointPlacement,jointName)
        self.model.appendBodyToJoint(jointId,inertia(.4,[0,0,0]),pin.SE3.Identity())
        self.gmodel.addGeometryObject( Capsule('world/thumb2',jointId,H,FL-0*H,
                                               pin.SE3(rotate('x',2*pi/3),np.array([-0.007*cm,.008*cm,-0.5*cm]))) )

        # Prepare some patches to represent collision points. Yet unvisible.
        if self.viewer is not None:
            self.maxContact = 10
            for i in range(self.maxContact):
                self.viewer.addCylinder('world/cpatch%d'%i, .01, .003, [ 1.0,0,0,1])
                self.viewer.setVisibility('world/cpatch%d'%i,'OFF')
    
    def hideContact(self,fromContactRef=0):
        if fromContactRef<0: fromContactRef=self.maxContact+fromContactRef
        for i in range(fromContactRef,self.maxContact):
            name='world/cpatch%d'%i
            self.viewer.setVisibility(name,'OFF')

    def displayContact(self,contact,contactRef=0,refresh=False):
        '''
        Display a small red disk at the position of the contact, perpendicular to the
        contact normal. 
        
        @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
        geomModel.collisionResults[0].geContact(0).
        @param contactRef: use patch named "world/cparch%d" % contactRef, 0 by default.
        @param refresh: option to refresh the viewer before returning, False by default.
        '''
        name='world/cpatch%d'%contactRef
        if self.viewer is None: return
        self.viewer.setVisibility(name,'ON')
        M = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0,0,1]),contact.normal).matrix(),contact.pos)
        self.viewer.applyConfiguration(name,pin.se3ToXYZQUATtuple(M))
        if refresh: self.viewer.refresh()   
        
    def display(self,q):
        pin.forwardKinematics(self.model,self.data,q)
        pin.updateGeometryPlacements(self.model,self.data,self.gmodel,self.gdata)
        if self.viewer is None: return
        for i,g in enumerate(self.gmodel.geometryObjects):
            self.viewer.applyConfiguration(g.name, pin.se3ToXYZQUATtuple(self.gdata.oMg[i]))
        self.viewer.refresh()



if __name__ == "__main__":
    from utils.meshcat_viewer_wrapper import MeshcatVisualizer
    import time
    import numpy as np
    from numpy import cos,sin
    robot = RobotHand()

    viz = MeshcatVisualizer(robot,url='classical')
    viz.display(robot.q0)

    q = robot.q0.copy()
    T = 1000
    for t in range(T):
        x = sin(t/30)
        q[12:13] = x+.5
        viz.display(q)
        time.sleep(5e-2)
