import pinocchio as pin
import numpy as np

class CollisionWrapper:
    def __init__(self,robot,viz=None):
        self.robot=robot
        self.viz=viz
        
        self.rmodel = robot.model
        self.rdata  = self.rmodel.createData()
        self.gmodel = self.robot.gmodel
        self.gdata  = self.gmodel.createData()
        self.gdata.collisionRequests.enable_contact = True


    def computeCollisions(self,q,vq=None):
        res = pin.computeCollisions(self.rmodel,self.rdata,self.gmodel,self.gdata,q,False)
        pin.computeDistances(self.rmodel,self.rdata,self.gmodel,self.gdata,q)
        pin.computeJointJacobians(self.rmodel,self.rdata,q)
        if vq is not None:
            pin.forwardKinematics(self.rmodel,self.rdata,q,vq,0*vq)
        return res

    def getCollisionList(self):
        '''Return a list of triplets [ index,collision,result ] where index is the
        index of the collision pair, colision is gmodel.collisionPairs[index]
        and result is gdata.collisionResults[index].
        '''
        return [ [ir,self.gmodel.collisionPairs[ir],r]
                 for ir,r in enumerate(self.gdata.collisionResults) if r.isCollision() ]

    def _getCollisionJacobian(self,col,res):
        '''Compute the jacobian for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0,0,1]),contact.normal).matrix(),contact.pos)
    
        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse()*oMj1
        cMj2 = oMc.inverse()*oMj2
        
        J1=pin.getJointJacobian(self.rmodel,self.rdata,joint1,pin.ReferenceFrame.LOCAL)
        J2=pin.getJointJacobian(self.rmodel,self.rdata,joint2,pin.ReferenceFrame.LOCAL)
        Jc1=cMj1.action@J1
        Jc2=cMj2.action@J2
        J = (Jc2-Jc1)[2,:]
        return J

    def _getCollisionJdotQdot(self,col,res):
        '''Compute the Coriolis acceleration for one collision only. '''
        contact = res.getContact(0)
        g1 = self.gmodel.geometryObjects[col.first]
        g2 = self.gmodel.geometryObjects[col.second]
        oMc = pin.SE3(pin.Quaternion.FromTwoVectors(np.array([0,0,1]),contact.normal).matrix(),contact.pos)
    
        joint1 = g1.parentJoint
        joint2 = g2.parentJoint
        oMj1 = self.rdata.oMi[joint1]
        oMj2 = self.rdata.oMi[joint2]

        cMj1 = oMc.inverse()*oMj1
        cMj2 = oMc.inverse()*oMj2

        a1 = self.rdata.a[joint1]
        a2 = self.rdata.a[joint2]
        a = (cMj1 * a1 - cMj2 * a2).linear[2]
        return a
        
    def getCollisionJacobian(self,collisions=None):
        '''From a collision list, return the Jacobian corresponding to the normal direction.  '''
        if collisions is None: collisions = self.getCollisionList()
        if len(collisions)==0: return np.ndarray([0,self.rmodel.nv])
        J = np.vstack([ self._getCollisionJacobian(c,r) for (i,c,r) in collisions ])
        return J

    def getCollisionJdotQdot(self,collisions=None):
        if collisions is None:
            collisions = self.getCollisionList()
        if len(collisions)==0:
            return np.array([])
        a0 = np.array([self._getCollisionJdotQdot(c,r) for (i,c,r) in collisions])
        return a0

    def getCollisionDistances(self,collisions=None):
        if collisions is None: collisions = self.getCollisionList()
        if len(collisions)==0: return np.array([])
        dist = np.array([ self.gdata.distanceResults[i].min_distance for (i,c,r) in collisions ])
        return dist


    # --- DISPLAY -----------------------------------------------------------------------------------
    # --- DISPLAY -----------------------------------------------------------------------------------
    # --- DISPLAY -----------------------------------------------------------------------------------

    def initDisplay(self,viz=None):
        if viz is not None: self.viz = viz
        assert(self.viz is not None)

        self.patchName = 'world/contact_%d_%s'
        self.ncollisions=10
        self.createDisplayPatchs(0)
        
    def createDisplayPatchs(self,ncollisions):

        if ncollisions == self.ncollisions: return
        elif ncollisions<self.ncollisions: # Remove patches
            for i in range(ncollisions,self.ncollisions):
                viz[self.patchName % (i,'a')].delete()
                viz[self.patchName % (i,'b')].delete()
        else:
            for i in range(self.ncollisions,ncollisions):
                viz.addCylinder( self.patchName % (i,'a') , .0005,.005,"red") 
                #viz.addCylinder( self.patchName % (i,'b') , .0005,.05,"red") 
        
        self.ncollisions = ncollisions

    def displayContact(self,ipatch,contact):
        '''
        Display a small red disk at the position of the contact, perpendicular to the
        contact normal. 
        
        @param ipatchf: use patch named "world/contact_%d" % contactRef.
        @param contact: the contact object, taken from Pinocchio (HPP-FCL) e.g.
        geomModel.collisionResults[0].geContact(0).
        '''
        name = self.patchName % (ipatch, 'a')
        R = pin.Quaternion.FromTwoVectors(np.array([0,1,0]),contact.normal).matrix()
        M = pin.SE3(R,contact.pos)
        self.viz.applyConfiguration(name,M)

        
    def displayCollisions(self,collisions = None):
        '''Display in the viewer the collision list get from getCollisionList().'''
        if self.viz is None: return
        if collisions is None: collisions = self.getCollisionList()

        self.createDisplayPatchs(len(collisions))
        for ic,[i,c,r] in enumerate(collisions):
            self.displayContact(ic,r.getContact(0))


if __name__ == "__main__":
    from utils.meshcat_viewer_wrapper import MeshcatVisualizer
    from tp4.robot_hand import RobotHand
    import time
    import numpy as np
    from numpy import cos,sin
    np.random.seed(3)
    
    robot = RobotHand()

    viz = MeshcatVisualizer(robot,url='classical')

    q = robot.q0.copy()
    q[0]=.5
    q[2:4]=1.7648
    viz.display(q)

    col = CollisionWrapper(robot,viz)
    col.initDisplay()
    col.createDisplayPatchs(1)
    col.computeCollisions(q)
    cols = col.getCollisionList()

    ci=cols[0][2]     
    col.displayContact(0,ci.getContact(0))   

    ### Try to find a random contact
    if 0:
        q = robot.q0.copy()
        vq = np.random.rand(robot.model.nv)*2-1
        for i in range(10000):
            q+=vq*1e-3
            col.computeCollisions(q)
            cols = col.getCollisionList() 
            if len(cols)>0: break
            if not i % 20: viz.display(q)
            
        viz.display(q)
    
        col.displayCollisions()
        p = cols[0][1]
        ci = cols[0][2].getContact(0)
        
        import pickle
        with open('/tmp/bug.pickle', 'wb') as file:
            pickle.dump([ col.gdata.oMg[11],
                          col.gdata.oMg[3],
                          #col.gmodel.geometryObjects[11].geometry,
            ] , file)

    dist=col.getCollisionDistances()
    J = col.getCollisionJacobian()
