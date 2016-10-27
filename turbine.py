# -*- coding: utf-8 -*-
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import util
from math import radians, pi
from spline import Spline, SplineError, Arc

class SuctionSide(Spline):
    
    def __init__(self, parameters, tMethod, derMethod, dertMethod, vs3Method, der2, lambd):
        
        beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
        
        # Convert the angles to radians
        beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
        dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
        GAMMA = radians(GAMMA)
        
        chord = (Bx - R1*(1 - sp.cos(beta1)) - R2*(1 - sp.cos(beta2)))/sp.cos(gamma)
        height = chord*sp.sin(gamma) + R1*sp.sin(beta1) - R2*sp.sin(beta2)
        
        s1 = np.array((R1*(1 - sp.sin(dbeta1/2 + beta1)), height + R1*sp.cos(dbeta1/2 + beta1)))
        s3 = np.array((Bx - (tau*sp.cos(betag) + R2)*sp.sin(beta2-GAMMA) - R2, tau - (tau*sp.cos(betag) + R2)*sp.cos(beta2-GAMMA)))
        s5 = np.array((Bx - R2*(1 - sp.sin(beta2 + dbeta2/2)), R2*sp.cos(beta2 + dbeta2/2)))
        
        s2 = np.array(((s1[0]+s3[0])/2,lambd*(s1[0]+s3[0])/2))
        
        p1 = (R1*(1 + sp.sin(beta1 - dbeta1/2.)), height - R1*sp.cos(beta1 - dbeta1/2.))
        p4 = (Bx - R2*(1 + sp.sin(beta2 - dbeta2/2.)), -R2*sp.cos(beta2 - dbeta2/2.))
        
        newparam = util.spacing(np.array([p1, s1, s2, s3, s5, p4]), (0, 1), *dertMethod)
        tParam = util.spacing(np.array([s1, s2, s3, s5]), (0, 1), *tMethod)
        
        if der2 == True:
            tParam9 = [0,0,0,tParam[1],tParam[2],tParam[2],1,1,1]
        else:
            tParam9 = [0,0,tParam[1],tParam[2],tParam[2],1,1]
        knots = util.knotAveraging(tParam9, 5, (0,1))
        
        if derMethod == 'radius':
            vs1 = R1*np.array((sp.cos(dbeta1/2 + beta1), sp.sin(dbeta1/2 + beta1)))
            vs5 = R2*np.array((sp.cos(dbeta2/2 + beta2), -sp.sin(dbeta2/2 + beta2)))
            
        elif derMethod == 'fastparam':
            h1, h2 = newparam[1]-newparam[0], newparam[-1]-newparam[-2]
            vs1 = R1*(np.pi-beta1)/h1*np.array((sp.cos(dbeta1/2 + beta1), sp.sin(dbeta1/2 + beta1)))
            vs5 = R2*(np.pi-beta2)/h2*np.array((sp.cos(dbeta2/2 + beta2), -sp.sin(dbeta2/2 + beta2)))
            
        elif derMethod == 'param':
            h1, h2 = newparam[1]-newparam[0], newparam[-1]-newparam[-2]
            vs1 = R1*(np.pi-beta1)/h1/5.*np.array((sp.cos(dbeta1/2 + beta1), sp.sin(dbeta1/2 + beta1)))
            vs5 = R2*(np.pi-beta2)/h2/5.*np.array((sp.cos(dbeta2/2 + beta2), -sp.sin(dbeta2/2 + beta2)))
        else:
            raise SplineError, "derMethod must be one of 'radius', 'fastparam' or 'param'."
        
        if type(vs3Method) in (int, float):
            vs3 = vs3Method*np.array([sp.cos(beta2 - GAMMA),-sp.sin(beta2 - GAMMA)])
        elif vs3Method == 'average':
            leng = (1 - tParam[2])*np.linalg.norm(vs1) + tParam[2]*np.linalg.norm(vs5)
            vs3 = leng*np.array([sp.cos(beta2 - GAMMA),-sp.sin(beta2 - GAMMA)])
        else:
            raise SplineError, "vs3Method must be either a number or 'average'."
        
        if der2 == True:
            as1 = np.array((sp.sin(dbeta1/2 + beta1), -sp.cos(dbeta1/2 + beta1)))/20.
            as5 = np.array((-sp.sin(dbeta2/2 + beta2), -sp.cos(dbeta2/2 + beta2)))/20.
        
        if der2 == True:
            conditions = np.array([s1,vs1,as1,s2,s3,vs3,s5,vs5,as5])
        else:
            conditions = np.array([s1,vs1,s2,s3,vs3,s5,vs5])
        
        basislist = util.basisarray(5, knots)
        der0 = basislist[-1]
        der1 = util.basisderivative(1, basislist, knots)
        
        if der2 == True:
            der2 = util.basisderivative(2, basislist, knots)
            rang = 9
            probmatrix = np.array([
            [der0[i](tParam[0]) for i in range(rang)],
            [der1[i](tParam[0]) for i in range(rang)],
            [der2[i](tParam[0]) for i in range(rang)],
            [der0[i](tParam[1]) for i in range(rang)],
            [der0[i](tParam[2]) for i in range(rang)],
            [der1[i](tParam[2]) for i in range(rang)],
            [der0[i](tParam[3]) for i in range(rang)],
            [der1[i](tParam[3]) for i in range(rang)],
            [der2[i](tParam[3]) for i in range(rang)],
            ])
        else:
            rang = 7
            probmatrix = np.array([
            [der0[i](tParam[0]) for i in range(rang)],
            [der1[i](tParam[0]) for i in range(rang)],
            [der0[i](tParam[1]) for i in range(rang)],
            [der0[i](tParam[2]) for i in range(rang)],
            [der1[i](tParam[2]) for i in range(rang)],
            [der0[i](tParam[3]) for i in range(rang)],
            [der1[i](tParam[3]) for i in range(rang)],
            ])
        
        xpts = np.linalg.solve(probmatrix,conditions[:,0])
        ypts = np.linalg.solve(probmatrix,conditions[:,1])
        
        self.conditions = conditions
        self.parameters = (beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2)
        
        super(SuctionSide, self).__init__(np.array((xpts,ypts)), knots)
#        Spline.__init__(np.array((xpts,ypts)), knots)
    
    @classmethod
    def functionOfLambda(cls, parameters, tMethod, derMethod, dertMethod, vs3Method, der2):
        def suctionside(l):
            return cls(parameters, tMethod, derMethod, dertMethod, vs3Method, der2, l)
        return suctionside

class PressureSide(Spline):
    
    def __init__(self, parameters, derMethod, dertMethod, der2Method):
        
        beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
        
        beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
        dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
        GAMMA = radians(GAMMA)
        
        chord = (Bx - R1*(1 - sp.cos(beta1)) - R2*(1 - sp.cos(beta2)))/sp.cos(gamma)
        height = chord*sp.sin(gamma) + R1*sp.sin(beta1) - R2*sp.sin(beta2)
        
        s1 = np.array((R1*(1 - sp.sin(dbeta1/2 + beta1)), height + R1*sp.cos(dbeta1/2 + beta1)))
        s5 = np.array((Bx - R2*(1 - sp.sin(beta2 + dbeta2/2)), R2*sp.cos(beta2 + dbeta2/2)))
        
        p1 = (R1*(1 + sp.sin(beta1 - dbeta1/2.)), height - R1*sp.cos(beta1 - dbeta1/2.))
        p4 = (Bx - R2*(1 + sp.sin(beta2 - dbeta2/2.)), -R2*sp.cos(beta2 - dbeta2/2.))
        
        newparam = util.spacing(np.array((s5,p4,p1,s1)), (0,1), *dertMethod)
        tParam = (0,1)
        knots = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
        
        if derMethod == 'radius':
            vp1 = R1*np.array([-sp.cos(beta1 - dbeta1/2.), -sp.sin(beta1 - dbeta1/2.)])
            vp4 = R2*np.array([-sp.cos(beta2 - dbeta2/2.), sp.sin(beta2 - dbeta2/2.)])
        elif derMethod == 'fastparam':
            h2, h1 = newparam[1]-newparam[0], newparam[-1]-newparam[-2]
            vp1 = R1*(np.pi-beta1)/h1*np.array([-sp.cos(beta1 - dbeta1/2.), -sp.sin(beta1 - dbeta1/2.)])
            vp4 = R2*(np.pi-beta2)/h2*np.array([-sp.cos(beta2 - dbeta2/2.), sp.sin(beta2 - dbeta2/2.)])
        elif derMethod == 'param':
            h2, h1 = newparam[1]-newparam[0], newparam[-1]-newparam[-2]
            vp1 = R1*(np.pi-beta1)/h1/5.*np.array([-sp.cos(beta1 - dbeta1/2.), -sp.sin(beta1 - dbeta1/2.)])
            vp4 = R2*(np.pi-beta2)/h2/5.*np.array([-sp.cos(beta2 - dbeta2/2.), sp.sin(beta2 - dbeta2/2.)])
        else:
            raise SplineError, "derMethod must be one of 'radius', 'fastparam' or 'param'."
        
        if der2Method == 'constant':
            ap1 = np.array([1.,1.])
            ap4 = np.array([1.,1.])
        elif der2Method == 'radius':
            ap1 = R1/20.*np.array((sp.sin(beta1 - dbeta1/2.), -sp.cos(beta1 - dbeta1/2.)))
            ap4 = R2/20.*np.array((-sp.sin(beta2 - dbeta2/2.), -sp.cos(beta2 - dbeta2/2.)))
        else:
            raise SplineError, "der2Method must be 'constant' or 'radius'."
        
        conditions = np.array([p4, vp4, ap4, p1, vp1, ap1])
        
        basislist = util.basisarray(5, knots)
        der0 = basislist[-1]
        der1 = util.basisderivative(1, basislist, knots)
        der2 = util.basisderivative(2, basislist, knots)
        
        rang = 6
        probmatrix = np.array([
        [der0[i](tParam[0]) for i in range(rang)],
        [der1[i](tParam[0]) for i in range(rang)],
        [der2[i](tParam[0]) for i in range(rang)],
        [der0[i](tParam[1]) for i in range(rang)],
        [der1[i](tParam[1]) for i in range(rang)],
        [der2[i](tParam[1]) for i in range(rang)]
         ])
        
        xpts = np.linalg.solve(probmatrix,conditions[:,0])
        ypts = np.linalg.solve(probmatrix,conditions[:,1])
        
        self.conditions = conditions
        self.parameters = (beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2)
        
        super(PressureSide, self).__init__(np.array((xpts,ypts)), knots)

class Turbine(Spline):
    
    @classmethod
    def by_conditions(cls, beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2, lambd = 3.8, rads = False):
        if not rads:
            beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
            dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
            GAMMA = radians(GAMMA)
        
        chord = (Bx - R1*(1 - sp.cos(beta1)) - R2*(1 - sp.cos(beta2)))/sp.cos(gamma)
        height = chord*sp.sin(gamma) + R1*sp.sin(beta1) - R2*sp.sin(beta2)
        
        s1 = np.array((R1*(1 - sp.sin(dbeta1/2 + beta1)), height + R1*sp.cos(dbeta1/2 + beta1)))
        s3 = np.array((Bx - (tau*sp.cos(betag) + R2)*sp.sin(beta2-GAMMA) - R2, tau - (tau*sp.cos(betag) + R2)*sp.cos(beta2-GAMMA)))
        s5 = np.array((Bx - R2*(1 - sp.sin(beta2 + dbeta2/2)), R2*sp.cos(beta2 + dbeta2/2)))
        if not lambd == None:
            s2 = np.array(((s1[0]+s3[0])/2,lambd*(s1[0]+s3[0])/2))
        
        p1 = (R1*(1 + sp.sin(beta1 - dbeta1/2.)), height - R1*sp.cos(beta1 - dbeta1/2.))
        p4 = (Bx - R2*(1 + sp.sin(beta2 - dbeta2/2.)), -R2*sp.cos(beta2 - dbeta2/2.))
        if not lambd == None:
            newparam = util.spacing(np.array((p1,s1,s2,s3,s5,p4)), (0,1), 'chordlength')
            tParam = util.spacing(np.array((p1,s1,s2,s3,s5,p4)), (0,1), 'affineangle',1.5)
        else:
            newparam = util.spacing(np.array((p1,s1,s3,s5,p4)), (0,1), 'affinechord')
            tParam = util.spacing(np.array((s1,s1,s3,s5,s5)), (0,1), 'affineangle',1.5)
#        tParam = [0.,1./6,2./5,1.]
#        tParam = util.spacing(np.array((s1,s2,s3,s5)), (0,1), 'uniform')
        print tParam
        if not lambd == None:
            tParam9 = [0,0,0,tParam[1],tParam[2],tParam[2],1,1,1]
            knots = [0,0,0,0,0,0,1./5*sum(tParam9[1:6]),1./5*sum(tParam9[2:7]),1./5*sum(tParam9[3:8]),1,1,1,1,1,1]
#            knots = [0,0,0,0,0,0,.25,.5,.75,1,1,1,1,1,1]
        else:
            tParam9 = [0,0,0,tParam[1],tParam[1],1,1,1]
            knots = [0,0,0,0,0,0,1./5*sum(tParam9[1:6]),1./5*sum(tParam9[2:7]),1,1,1,1,1,1]
#            knots = [0,0,0,0,0,0,1./3,2./3,1,1,1,1,1,1]
#        knots = [0,0,0,0,0,0,0.5,1,1,1,1,1,1]
#        knots = [0,0,0,0,1./3*sum(tParam9[1:4]),1./3*sum(tParam9[2:5]),1./3*sum(tParam9[3:6]),1./3*sum(tParam9[4:7]),1./3*sum(tParam9[5:8]),1,1,1,1]
        h1, h2 = newparam[1]-newparam[0], newparam[-1]-newparam[-2]
        v1 = R1*(np.pi-beta1)/h1/5*np.array((sp.cos(dbeta1/2 + beta1), sp.sin(dbeta1/2 + beta1)))
        print R1*(np.pi-beta1)/h1/5
        if not lambd == None:
            v2 = ((1-tParam[2])*R1*(np.pi-beta1)/h1 + tParam[2]*R2*(np.pi-beta2)/h2)/5*np.array([sp.cos(beta2 - GAMMA),-sp.sin(beta2 - GAMMA)])
        else:
            v2 = ((1-tParam[1])*R1*(np.pi-beta1)/h1 + tParam[1]*R2*(np.pi-beta2)/h2)/5*np.array([sp.cos(beta2 - GAMMA),-sp.sin(beta2 - GAMMA)])
        v3 = R2*(np.pi-beta2)/h2/5*np.array((sp.cos(dbeta2/2 + beta2), -sp.sin(dbeta2/2 + beta2)))
#        a1 = np.array((0,0))
#        a2 = np.array((0,0))
        
        tmpa1 = (R1*(np.pi-beta1)/h1/5)**2*np.array((sp.sin(dbeta1/2 + beta1), -sp.cos(dbeta1/2 + beta1)))
        tmpa2 = (R2*(np.pi-beta2)/h2/5)**2*np.array((-sp.sin(dbeta2/2 + beta2), -sp.cos(dbeta2/2 + beta2)))
        
        paral1 = lambda t: tmpa1 + t*v1
        paral2 = lambda t: tmpa2 + t*v3
        
        a1 = paral1(0)
        a2 = paral2(0)
            
        """
        v1 = R1*(np.pi-beta1)/(R1)*np.array((sp.cos(dbeta1/2 + beta1), sp.sin(dbeta1/2 + beta1)))
        v2 = ((1-tParam[2])*R1*(np.pi-beta1)/(R1) + tParam[2]*R2*(np.pi-beta2)/(R2))*np.array([sp.cos(beta2 - GAMMA),-sp.sin(beta2 - GAMMA)])
        v3 = R2*(np.pi-beta2)/(R2)*np.array((sp.cos(dbeta2/2 + beta2), -sp.sin(dbeta2/2 + beta2)))
        a1 = R1*((np.pi-beta1)/(R1))**2*np.array((sp.sin(dbeta1/2 + beta1), -sp.cos(dbeta1/2 + beta1)))
        a2 = R2*((np.pi-beta2)/(R2))**2*np.array((-sp.sin(dbeta2/2 + beta2), -sp.cos(dbeta2/2 + beta2)))
        """
        if not lambd == None:
            conditions = np.array([s1,v1,a1,s2,s3,v2,s5,v3,a2])
        else:
            conditions = np.array([s1,v1/5.,a1/20.,s3,v2/5.,s5,v3/5.,a2/20.])
#        conditions = np.array([s1,v1/5.,a1/20.,s3,s5,v3/5.,a2/20.])
        
        basislist = util.basisarray(5, knots)
        der0 = basislist[-1]
        der1 = util.basisderivative(1, basislist, knots)
        der2 = util.basisderivative(2, basislist, knots)
        
        if not lambd == None:
            rang = 9
            probmatrix = np.array([
            [der0[i](tParam[0]) for i in range(rang)],
            [der1[i](tParam[0]) for i in range(rang)],
            [der2[i](tParam[0]) for i in range(rang)],
            [der0[i](tParam[1]) for i in range(rang)],
            [der0[i](tParam[2]) for i in range(rang)],
            [der1[i](tParam[2]) for i in range(rang)],
            [der0[i](tParam[3]) for i in range(rang)],
            [der1[i](tParam[3]) for i in range(rang)],
            [der2[i](tParam[3]) for i in range(rang)],
            ])
        else:
            rang = 8
            probmatrix = np.array([
            [der0[i](tParam[0]) for i in range(rang)],
            [der1[i](tParam[0]) for i in range(rang)],
            [der2[i](tParam[0]) for i in range(rang)],
            [der0[i](tParam[1]) for i in range(rang)],
            [der1[i](tParam[1]) for i in range(rang)],
            [der0[i](tParam[2]) for i in range(rang)],
            [der1[i](tParam[2]) for i in range(rang)],
            [der2[i](tParam[2]) for i in range(rang)],
            ])
        
        xpts = np.linalg.solve(probmatrix,conditions[:,0])
        ypts = np.linalg.solve(probmatrix,conditions[:,1])
        
        return cls(np.array((xpts,ypts)), knots), conditions

#==============================================================================
# def leadingEdge(parameters):
#     """
#     Returns an arc that goes from p1 to s1 as x goes from 0 to 1.
#     """
#     beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
#         
#     beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
#     dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
#     GAMMA = radians(GAMMA)
#     
#     chord = (Bx - R1*(1 - sp.cos(beta1)) - R2*(1 - sp.cos(beta2)))/sp.cos(gamma)
#     height = chord*sp.sin(gamma) + R1*sp.sin(beta1) - R2*sp.sin(beta2)
#     
#     def arc(x):
#         param = lambda x: pi + beta1 + dbeta1/2. - x*(pi/2. + 2*beta1 + dbeta1)
#         return np.array((R1*(1 - np.cos(param(x))), height + R1*np.sin(param(x))))
#     
#     return arc
# 
# def trailingEdge(parameters):
#     """
#     Returns an arc that goes from s5 to p4 as x goes from 0 to 1.
#     """
#     beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
#         
#     beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
#     dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
#     GAMMA = radians(GAMMA)
#     
#     def arc(x):
#         param = lambda x: pi/2. - beta2 - dbeta2/2. + x*dbeta2
#         return np.array((Bx - R2*(1 - np.cos(param(x))), R2*np.sin(param(x))))
#    
#     return arc
#==============================================================================

def leadingEdge(parameters):
    beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
    
    beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
    dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
    GAMMA = radians(GAMMA)
    
    chord = (Bx - R1*(1 - sp.cos(beta1)) - R2*(1 - sp.cos(beta2)))/sp.cos(gamma)
    height = chord*sp.sin(gamma) + R1*sp.sin(beta1) - R2*sp.sin(beta2)
    
    leadingedge = Arc(R1, (R1, height), pi/2. + beta1 + dbeta1/2., beta1 - pi/2. - dbeta1/2.)
    return leadingedge

def trailingEdge(parameters):
    beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = parameters
    
    beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
    dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
    GAMMA = radians(GAMMA)
    
    trailingedge = Arc(R2, (Bx - R2,0), dbeta2/2. - pi/2. - beta2, pi/2. - beta2 - dbeta2/2.)
    return trailingedge
    
if __name__ == "__main__":
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    suctionside = SuctionSide.functionOfLambda(inputData, ('affineangle',1.5), 'param', ('affinechord',), 'average', False)
    pressureside = PressureSide(inputData, 'param', ('centripedal',0.9), 'radius')
    pressureside.plot2D(100, False)
#    plt.show()
    leadingedge = leadingEdge(inputData)
    trailingedge = trailingEdge(inputData)
#    tvals = np.linspace(0,1,201)
#    vals = np.array([leadingedge(t) for t in tvals])
#    print vals
#    plt.plot(vals.T[0],vals.T[1])
#    fig = plt.figure()
#    fig.set_aspect(1)
    leadingedge.plot2D(100, False)
    trailingedge.plot2D(100, False)
#    plt.show()
    print(suctionside)
    suctionside(3.4).plot2D(100, False)
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()