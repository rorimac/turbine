# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from Nurbs import Crv
from spline import Spline
from turbine import SuctionSide, PressureSide, Turbine, leadingEdge, trailingEdge
import util
from math import radians, pi
from matplotlib.lines import Line2D

def suctionside(inputData):
    turbine, conditions = Turbine.by_conditions(*inputData,lambd = 3.5)
    newconditions = np.array([conditions[0],np.array([0.22,0.94]),np.array([0.5,.97]),np.array([0.77,.7]),conditions[6]])
#    plt.plot(newconditions[:,0], newconditions[:,1])
#    plt.show()
    tvals = util.spacing(newconditions, (0,1), 'uniform')
    knots = np.array([0,0,0,0, 1/3.*sum(tvals[1:4]), 1,1,1,1])
    basis = util.basisarray(3,knots)
    der0 = basis[-1]
    rang = 5
    probmatrix = np.array([
    [der0[i](tvals[0]) for i in range(rang)],
    [der0[i](tvals[1]) for i in range(rang)],
    [der0[i](tvals[2]) for i in range(rang)],
    [der0[i](tvals[3]) for i in range(rang)],
    [der0[i](tvals[4]) for i in range(rang)],
    ])
    
    xpts = np.linalg.solve(probmatrix,newconditions[:,0])
    ypts = np.linalg.solve(probmatrix,newconditions[:,1])
    
    return Spline(np.array([xpts,ypts]),knots)

def pressureside(inputData):
    beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = inputData
    
    beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
    dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
    GAMMA = radians(GAMMA)
    
    chord = (Bx - R1*(1 - np.cos(beta1)) - R2*(1 - np.cos(beta2)))/np.cos(gamma)
    height = chord*np.sin(gamma) + R1*np.sin(beta1) - R2*np.sin(beta2)
    
    p1 = (R1*(1 + np.sin(beta1 - dbeta1/2.)), height - R1*np.cos(beta1 - dbeta1/2.))
    p4 = (Bx - R2*(1 + np.sin(beta2 - dbeta2/2.)), -R2*np.cos(beta2 - dbeta2/2.))
    newconditions = np.array([p1, np.array([0.3,0.57]), np.array([0.61,0.48]), p4])

    tvals = util.spacing(newconditions, (0, 1), 'uniform')
    knots = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    basis = util.basisarray(3, knots)
    der0 = basis[-1]
    rang = 4
    probmatrix = np.array([
    [der0[i](tvals[0]) for i in range(rang)],
    [der0[i](tvals[1]) for i in range(rang)],
    [der0[i](tvals[2]) for i in range(rang)],
    [der0[i](tvals[3]) for i in range(rang)],
    ])
    
    xpts = np.linalg.solve(probmatrix,newconditions[:,0])
    ypts = np.linalg.solve(probmatrix,newconditions[:,1])
    
    return Spline(np.array([xpts,ypts]),knots)

def arc(pt1, pt2, pt3, pt4, alpha):
    length = np.linalg.norm(pt2-pt3)/alpha
    v1 = (pt2 - pt1)/np.linalg.norm(pt2-pt1)
    v2 = (pt3 - pt4)/np.linalg.norm(pt3-pt4)
    newpt1 = pt2 + v1*length
    newpt2 = pt3 + v2*length
    ary = np.array([pt2, newpt1, newpt2, pt3]).T
    return Spline(ary, np.array([0,0,0,0,1,1,1,1]))
    

def geometryplot(shw = False):
#                beta1     beta2  Gamma   tau   betag  gamma  Bx    R1      R2      dbeta1 dbeta2
    inputData = [90-44.5, 90-21.8, 14.8, .825, 24.4424, 29.2, 1., .0662, .0228, 27.5, 6.5]
    beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = inputData
    
    beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
    dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
    GAMMA = radians(GAMMA)
    chord = (Bx - R1*(1 - np.cos(beta1)) - R2*(1 - np.cos(beta2)))/np.cos(gamma)
    height = chord*np.sin(gamma) + R1*np.sin(beta1) - R2*np.sin(beta2)
    
    curve1 = suctionside(inputData)
    curve2 = pressureside(inputData)
    arc1 = arc(curve1.cntrl[:2,1],curve1.cntrl[:2,0],curve2.cntrl[:2,0],curve2.cntrl[:2,1], 1.5)
    arc2 = arc(curve2.cntrl[:2,-2],curve2.cntrl[:2,-1],curve1.cntrl[:2,-1],curve1.cntrl[:2,-2], 1.4)
    n = 100
    pnts1 = curve1.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
    pnts2 = curve2.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
    pnts3 = arc1.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
    pnts4 = arc2.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
    
#    fig = plt.figure(figsize = (3*7.2727272727272727,3*10))
    tmp = 2.6
    fig = plt.figure(figsize = (tmp*7.2,tmp*10))
    ax1 = fig.add_subplot(121)
    ax1.axis([-.3, 1.3, -.2, 2])
    ax1.plot(pnts1[0], pnts1[1], 'k', label='parametric bspline')
    ax1.plot(pnts2[0], pnts2[1], 'k', label='parametric bspline')
    ax1.plot(pnts3[0], pnts3[1], 'k', label='parametric bspline')
    ax1.plot(pnts4[0], pnts4[1], 'k', label='parametric bspline')
    ax1.set_aspect(1)
    
    ax2 = fig.add_subplot(121)
    ax2.plot(pnts1[0], [pnts1[1][i] + inputData[3] for i in range(len(pnts1[1]))], 'k', label='parametric bspline')
    ax2.plot(pnts2[0], [pnts2[1][i] + inputData[3] for i in range(len(pnts2[1]))], 'k', label='parametric bspline')
    ax2.plot(pnts3[0], [pnts3[1][i] + inputData[3] for i in range(len(pnts3[1]))], 'k', label='parametric bspline')
    ax2.plot(pnts4[0], [pnts4[1][i] + inputData[3] for i in range(len(pnts4[1]))], 'k', label='parametric bspline')
    
    ax2.plot([-.3,R1],[height,height],'k.-.')
    
    #plotting x and y axis
    ax2.annotate("",
            xy=(-0.3,0), xycoords='data',
            xytext=(1.2, -0.0), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3"),
            )
    ax2.annotate("",
            xy=(0, -.2), xycoords='data',
            xytext=(-0.0, 1.8), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3"),
            )
    ax2.annotate(r'$x$', xy=(-0.15,height + tau/2), xytext=(1.16,-0.07),
            )
    ax2.annotate(r'$y$', xy=(-0.15,height + tau/2), xytext=(-0.07,1.77),
            )
    
    #plotting height
    ax2.annotate("",
            xy=(-0.15,0), xycoords='data',
            xytext=(-0.15, height), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"),
            )
    ax2.annotate(r'$H$', xy=(-0.15, height/2), xytext=(-.2, height/2.2),
            )
    
    #plotting tau
    ax2.plot([-.3,R1],[height+tau,height+tau],'k.-.')
    
    ax2.annotate("",
            xy=(-0.15,height), xycoords='data',
            xytext=(-0.15,tau + height), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3"),
            )
    ax2.annotate(r'$\tau$', xy=(-0.15,height + tau/2), xytext=(-.19,height + tau/2.2),
            )
    
    #plotting R1
    ax2.plot((R1,pnts1[0,0]), (tau + height, pnts1[1,0] + inputData[3]),'k-')
    
    ax2.annotate(r'$R_1$',
            xy=(R1,tau + height), xycoords='data',
            xytext=(pnts1[0,0]+0.02, pnts1[1,0] + inputData[3] - 0.01), textcoords='data',
            )
    
    #plotting R2
    ax2.plot((pnts1[0,-1],Bx-R2), (pnts1[1,-1]+tau,tau),'k-')
    
    ax2.annotate(r'$R_2$',
            xy=(R1,tau + height), xycoords='data',
            xytext=(pnts1[0,-1]-0.05, pnts1[1,-1] + inputData[3] + 0.01), textcoords='data',
            )
            
    #plotting beta1 and dbeta1
    tmp = 0.44
    ax2.plot([curve1.cntrl[0,0], curve1.cntrl[0,0]-tmp*(curve1.cntrl[0,1]-curve1.cntrl[0,0])], [tau + curve1.cntrl[1,0],tau + curve1.cntrl[1,0]-tmp*(curve1.cntrl[1,1]-curve1.cntrl[1,0])], 'k-')
    tmp = 1.1
    ax2.plot([curve2.cntrl[0,0], curve2.cntrl[0,0]-tmp*(curve2.cntrl[0,1]-curve2.cntrl[0,0])], [tau + curve2.cntrl[1,0],tau + curve2.cntrl[1,0]-tmp*(curve2.cntrl[1,1]-curve2.cntrl[1,0])], 'k-')
    ax2.plot([R1, -.1], [tau + height,tau + height - .1945], 'k-')
    ax2.annotate("",
            xy=(-0.12,height + tau), xycoords='data',
            xytext=(-.06,tau + height - .15), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.3"),
            )
    ax2.annotate(r'$\beta_1$', xy=(-0.15,height + tau/2), xytext=(-.1,tau + height - .09),
            )
    ax2.annotate("",
            xy=(curve1.cntrl[0,0]-0.25*(curve1.cntrl[0,1]-curve1.cntrl[0,0]),tau + curve1.cntrl[1,0]-0.25*(curve1.cntrl[1,1]-curve1.cntrl[1,0])), xycoords='data',
            xytext=(curve2.cntrl[0,0]-0.6*(curve2.cntrl[0,1]-curve2.cntrl[0,0]), tau + curve2.cntrl[1,0]-0.6*(curve2.cntrl[1,1]-curve2.cntrl[1,0])), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0.1"),
            )
    ax2.annotate(r'$\Delta\beta_1$', xy=(-0.15,height + tau/2), xytext=(.02,tau + height - .13),
            )
    
    #plotting beta2 and dbeta2
    tmp = 0.785
    ax2.plot([curve1.cntrl[0,-1], curve1.cntrl[0,-1]-tmp*(curve1.cntrl[0,-2]-curve1.cntrl[0,-1])], [tau + curve1.cntrl[1,-1],tau + curve1.cntrl[1,-1]-tmp*(curve1.cntrl[1,-2]-curve1.cntrl[1,-1])], 'k-')
    tmp = 0.6
    ax2.plot([curve2.cntrl[0,-1], curve2.cntrl[0,-1]-tmp*(curve2.cntrl[0,-2]-curve2.cntrl[0,-1])], [tau + curve2.cntrl[1,-1],tau + curve2.cntrl[1,-1]-tmp*(curve2.cntrl[1,-2]-curve2.cntrl[1,-1])], 'k-')
    ax2.plot([Bx-R2, 1.3], [tau,tau], 'k.-.')
    ax2.plot([Bx-R2, Bx+R2+0.019], [tau,tau  -.2], 'k-')
    ax2.annotate("",
            xy=(Bx + 2.7*R2,tau), xycoords='data',
            xytext=(Bx,tau - .08), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0.1"),
            )
    ax2.annotate(r'$\beta_2$', xy=(-0.15,height + tau/2), xytext=(Bx+1.5*R2,tau - .07),
            )
    ax2.annotate("",
            xy=(curve1.cntrl[0,-1]-0.785*(curve1.cntrl[0,-2]-curve1.cntrl[0,-1]),tau + curve1.cntrl[1,-1]-0.785*(curve1.cntrl[1,-2]-curve1.cntrl[1,-1])), xycoords='data',
            xytext=(curve2.cntrl[0,-1]-0.6*(curve2.cntrl[0,-2]-curve2.cntrl[0,-1]),tau + curve2.cntrl[1,-1]-0.6*(curve2.cntrl[1,-2]-curve2.cntrl[1,-1])), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.1"),
            )
    ax2.annotate(r'$\Delta\beta_2$', xy=(-0.15,height + tau/2), xytext=(Bx+5.5*R2,tau - .53),
            )
        
    #plotting Bx
    ax2.plot([Bx,Bx],[0.1,-.2], 'k-.')
    ax2.annotate("",
            xy=(0,-.07), xycoords='data',
            xytext=(Bx,-.07), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0"),
            )
    ax2.annotate(r'$B_x$', xy=(-0.15,height + tau/2), xytext=(Bx/2,-.11),
            )
    
    #plotting s3
    s3tmp = curve1.pnt3D([0.739])
    ax2.plot(s3tmp[0],s3tmp[1], 'ko')
    ax2.annotate(r'$s_3$', xy=(-0.15,height + tau/2), xytext=(Bx-0.225,tau-.125),
            )
    
    #plotting s1 and s5
    s1tmp = curve1.pnt3D([0])
    s5tmp = curve1.pnt3D([1])
    ax2.plot(s1tmp[0],s1tmp[1], 'ko')
    ax2.annotate(r'$s_1$', xy=(-0.15,height + tau/2), xytext=(0.4*R1,height+0.03),
            )
    ax2.plot(s5tmp[0],s5tmp[1], 'ko')
    ax2.annotate(r'$s_5$', xy=(-0.15,height + tau/2), xytext=(Bx,0.02),
            )
    
    #plotting p1 and p4
    p1tmp = curve2.pnt3D([0])
    p4tmp = curve2.pnt3D([1])
    ax2.plot(p1tmp[0],p1tmp[1], 'ko')
    ax2.annotate(r'$p_1$', xy=(-0.15,height + tau/2), xytext=(1.6*R1,height-0.09),
            )
    ax2.plot(p4tmp[0],p4tmp[1], 'ko')
    ax2.annotate(r'$p_4$', xy=(-0.15,height + tau/2), xytext=(Bx-0.07,-0.04),
            )
    
    #plotting s2
    s2tmp = curve1.pnt3D([0.4])
    ax2.plot(s2tmp[0],s2tmp[1], 'ko')
    ax2.annotate(r'$s_2$', xy=(-0.15,height + tau/2), xytext=(Bx-.6,tau+0.185),
            )
    
    #plotting C
    ax2.annotate("",
            xy=(0.3*R1,tau + height-0.9*R1), xycoords='data',
            xytext=(Bx-0.55*R2,tau -.025), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0"),
            )
    ax2.annotate(r'$C$', xy=(-0.15,height + tau/2), xytext=(Bx-.6,tau+0.31),
            )
    
    #plotting gamma
    ax2.plot([0.3,0.7],[tau+.25,tau+.25], 'k-.')
    ax2.annotate("",
            xy=(0.68,tau+.25), xycoords='data',
            xytext=(0.655,tau+.155), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0.15"),
            )
    ax2.annotate(r'$\gamma$', xy=(-0.15,height + tau/2), xytext=(0.679,tau+0.19),
            )
    
    #plotting o
    p4tmp = curve2.pnt3D([1])
    ax2.annotate("",
            xy=s3tmp[:2], xycoords='data',
            xytext=(p4tmp[0],p4tmp[1]+tau), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=0"),
            )
    ax2.annotate(r'$o$', xy=(-0.15,height + tau/2), xytext=(Bx-0.145,tau-0.08),
            )
    
    #plotting GAMMA
    derivs3 = curve1.nderiv(1)([0.78])
    ax2.plot(s3tmp[0],tau + s3tmp[1], 'ko')
    tmpbeg, tmpend = 0.1,0.23
    ax2.plot([s3tmp[0]-tmpbeg*derivs3[0],s3tmp[0]+tmpend*derivs3[0]],[tau+s3tmp[1] -tmpbeg*derivs3[1],tau+s3tmp[1]+tmpend*derivs3[1]], 'k-')
    ax2.plot([0.8,Bx+R2+0.019],[tau+height,tau  -.2], 'k-.')
    ax2.annotate("",
            xy=(Bx-2.34*R2,tau+.1), xycoords='data',
            xytext=(Bx+.087,tau+.18), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.1"),
            )
    ax2.annotate(r'$\Gamma$', xy=(-0.15,height + tau/2), xytext=(Bx+.03,tau+.105),
            )
    
    ax2.set_frame_on(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    
    
#    ax2.axis([-.3, 1.3, -.2, 2])
    if shw == False:
        fig.savefig('../../latex/img/parallell_blades.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

def arcsplot(shw = False):
#    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
#    inputData = [90-44.5, 90-21.8, 14.8, .825, 24.4424, 29.2, 1., .0662, .0228, 27.5, 6.5]
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    beta1, beta2, GAMMA, tau, betag, gamma, Bx, R1, R2, dbeta1, dbeta2 = inputData
    
    beta1 = radians(beta1); beta2 = radians(beta2);dbeta1 = radians(dbeta1)
    dbeta2 = radians(dbeta2); betag = radians(betag); gamma = radians(gamma)
    GAMMA = radians(GAMMA)
    chord = (Bx - R1*(1 - np.cos(beta1)) - R2*(1 - np.cos(beta2)))/np.cos(gamma)
    height = chord*np.sin(gamma) + R1*np.sin(beta1) - R2*np.sin(beta2)
    
#    suct_side, cond = Turbine.by_conditions(*inputData,lambd = None)
    suct_side = SuctionSide.functionOfLambda(inputData,('affineangle',), 'param', ('affinechord',), 'average', True)(3.4)
    pres_side = PressureSide(inputData, 'param', ('chordlength',), 'radius')
    circle1 = Crv.Circle(R1, [R1, height])
    circle2 = Crv.Circle(R2, [Bx - R2, 0])
    tmp = 2.6
    fig = plt.figure(figsize = (tmp*7.2727272727272727,tmp*10))
    ax1 = fig.add_subplot(121)
    ax1.axis([-.3, 1.3, -.2, 1.5])
    ax1.set_aspect(1)
    tvals = np.linspace(0,1,201)
    circ1 = circle1(tvals)[:2]
    circ2 = circle2(tvals)[:2]
    suct1 = suct_side(tvals[:20])[:2]
    pres1 = pres_side(tvals[-40:])[:2]
    suct2 = suct_side(tvals[-135:])[:2]
    suct3 = suct_side(tvals[-25:])[:2]
    pres2 = pres_side(tvals[:30])[:2]
    ax1.plot(circ1[0,:],circ1[1,:], 'k')
    ax1.plot(pres1[0,:],pres1[1,:], 'k')
    ax1.plot(suct1[0,:],suct1[1,:], 'k')
    
    ax1.plot(circ2[0,:],circ2[1,:], 'k')
    ax1.plot(pres2[0,:],pres2[1,:], 'k')
    ax1.plot(suct2[0,:],suct2[1,:], 'k')
    
    ax1.plot(circ2[0,:],circ2[1,:]+tau, 'k')
    ax1.plot(pres2[0,:],pres2[1,:]+tau, 'k')
    ax1.plot(suct3[0,:],suct3[1,:]+tau, 'k')
    
    #plotting x and y axis
    ax1.annotate("",
            xy=(-0.3,0), xycoords='data',
            xytext=(1.2, -0.0), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3"),
            )
    ax1.annotate("",
            xy=(0, -.2), xycoords='data',
            xytext=(-0.0, .8), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3"),
            )
    ax1.annotate(r'$x$', xy=(-0.15,height), xytext=(1.16,-0.07),
            )
    ax1.annotate(r'$y$', xy=(-0.15,height), xytext=(-0.07,.77),
            )
            
    line_y = lambda x, (x0,x1),(y0,y1): (x - x0)*(y1 - y0)/(x1 - x0) + y0
    sslope1 = lambda x: line_y(x ,suct1[0,:2],suct1[1,:2])
    pslope1 = lambda x: line_y(x ,pres1[0,-2:],pres1[1,-2:])
    mslope1 = lambda x: line_y(x, (-0.08,R1),(pslope1(-0.08),height))
    ax1.plot(np.linspace(-.08,0,101),sslope1(np.linspace(-.08,0,101)),'k-.')
    ax1.plot(np.linspace(-.08,0.1,101),pslope1(np.linspace(-.08,.1,101)),'k-.')
    ax1.plot(np.linspace(-.08,R1,101),mslope1(np.linspace(-.08,R1,101)),'k-.')
    ax1.plot(np.linspace(-.18,R1,101), height*np.ones(101), 'k-.')
    
    #plotting s1 and s5
    s1tmp = suct_side.pnt3D([0])
    s5tmp = suct_side.pnt3D([1])
    ax1.plot(s1tmp[0],s1tmp[1], 'ko')
    ax1.annotate(r'$s_1$', xy=(-0.15,0), xytext=(0.4*R1,height+0.03),
            )
    ax1.plot(s5tmp[0],s5tmp[1], 'ko')
    ax1.annotate(r'$s_5$', xy=(-0.15,0), xytext=(Bx,0.02),
            )
    
    #plotting p1 and p4
    p1tmp = pres_side.pnt3D([1])
    p4tmp = pres_side.pnt3D([0])
    ax1.plot(p1tmp[0],p1tmp[1], 'ko')
    ax1.annotate(r'$p_1$', xy=(-0.15,0), xytext=(1.83*R1,height-0.05),
            )
    ax1.plot(p4tmp[0],p4tmp[1], 'ko')
    ax1.annotate(r'$p_4$', xy=(-0.15,0), xytext=(Bx-0.07,-0.04),
            )
    
    
    #plotting theta1, theta2
    theta1line = lambda x: line_y(x, (s1tmp[0],R1), (s1tmp[1],height))
    ax1.plot(R1,height,'k.')
    ax1.plot(np.linspace(-.18,R1,101), theta1line(np.linspace(-.18,R1,101)), 'k-')
    ax1.annotate("",
            xy=(-0.12, theta1line(-0.12)), xycoords='data',
            xytext=(-0.137, height), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-0.1"),
            )
    ax1.annotate(r'$\theta_s$', xy=(-0.15,0), xytext=(-0.18, height + 0.036),
            )
            
    thetaarc = Crv.Arc(1.7*R1, [R1, height],-0.8,0.95*pi)
    arc1 = thetaarc(tvals)[:2]
    ax1.plot(arc1[0,:],arc1[1,:], 'k')
    
    theta2line = lambda x: line_y(x, (R1,p1tmp[0]), (height,p1tmp[1]))
    ax1.plot(np.linspace(R1,0.23,101), theta2line(np.linspace(R1,0.23,101)), 'k-')
    ax1.annotate("",
            xy=(arc1[0,-1], height + 0.01), xycoords='data',
            xytext=(-0.04, height), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3,rad=.0"),
            )
    ax1.annotate("",
            xy=(arc1[0,0], arc1[1,0]), xycoords='data',
            xytext=(0.113-0.005, theta2line(0.113)-0.005), textcoords='data',
            arrowprops=dict(arrowstyle="<-",
                            connectionstyle="arc3,rad=.0"),
            )
    ax1.annotate(r'$\theta_p$', xy=(-0.15,0), xytext=(0.12, height + 0.09),
            )
    thetaarc = Crv.Arc(1.7*R1, [R1, height],-0.8,0.95*pi)
    arc1 = thetaarc(tvals)[:2]
    ax1.plot(arc1[0,:],arc1[1,:], 'k')
    
    #plotting s3 and its angles
    s3tmp = suct_side.pnt3D([0.617])
    ax1.plot(s3tmp[0],s3tmp[1], 'ko')
    ax1.annotate(r'$s_3$', xy=(-0.15,height + tau/2), xytext=(Bx-0.375,tau-.125),
            )
    ax1.plot((s3tmp[0],Bx-R2),(s3tmp[1],tau), 'k-')
    ax1.plot((s3tmp[0],Bx-R2),(s3tmp[1],s3tmp[1]), 'k-')
    ax1.plot((Bx-R2,Bx-R2),(s3tmp[1],tau), 'k-')
    ax1.annotate(r'$A$', xy=(-0.15,height + tau/2), xytext=(Bx-0.175,tau-.155),
            )
    ax1.annotate(r'$B$', xy=(-0.15,height + tau/2), xytext=(Bx-0.02,tau-.069),
            )
    ax1.annotate("",
            xy=(Bx - 0.14,s3tmp[1]), xycoords='data',
            xytext=(Bx - 0.15,s3tmp[1]+0.07), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3,rad=-.1"),
            )
    ax1.annotate(r'$\pi/2-(\beta_2-\Gamma)$', xy=(Bx - 0.14,s3tmp[1]+0.04), xytext=(Bx+0.06,tau-.089),
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3,rad=-.1"),
            )
    
    ax1.set_frame_on(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.axes.get_xaxis().set_visible(False)
    
    if not shw:
        fig.savefig('../../latex/img/angle_blades.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    
def diff_blade_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    turbine1 = SuctionSide.functionOfLambda(inputData,('affineangle',), 'param', ('affinechord',), 'average', False)
    turbine2 = PressureSide(inputData, 'param', ('chordlength',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor='white')
    turbine1(3.2).plot2D(100, False, fig, False, 'k-.')
    turbine1(3.3).plot2D(100, False, fig, False, 'k:')
    turbine1(3.4).plot2D(100, False, fig, False, 'k--')
    turbine2.plot2D(100, False, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    ax = plt.gca()
    ax.axis([-0.01,1.01,-0.1,1.3])
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_aspect('equal')
    if not shw:
        plt.savefig('../../latex/img/diff_blade_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()
    
def simp_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 4.4#;SuctionSide.functionOfLambda(inputData, ('affineangle',1.5), 'param', ('affinechord',), 'average', False)
    turbine1 = SuctionSide.functionOfLambda(inputData,('uniform',), 'radius', ('uniform',), 1, False)
    turbine2 = PressureSide(inputData, 'radius', ('uniform',), 'constant')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[2].label1.set_visible(False)
    yticks[4].label1.set_visible(False)
    yticks[6].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (ymin, ymin), (xmin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/simplistic_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()

def av_v2_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 3.4
    turbine1 = SuctionSide.functionOfLambda(inputData,('uniform',), 'radius', ('uniform',), 'average', False)
    turbine2 = PressureSide(inputData, 'radius', ('uniform',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[3].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    xticks[5].label1.set_visible(False)
    xticks[7].label1.set_visible(False)
    xticks[8].label1.set_visible(False)
    xticks[9].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[1].label1.set_visible(False)
    yticks[3].label1.set_visible(False)
    yticks[5].label1.set_visible(False)
    yticks[7].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/average_v2_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()

def av_v2_diff_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 3.4
    turbine1 = SuctionSide.functionOfLambda(inputData,('centripedal',), 'radius', ('uniform',), 'average', False)
    turbine2 = PressureSide(inputData, 'radius', ('chordlength',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[2].label1.set_visible(False)
    yticks[4].label1.set_visible(False)
    yticks[6].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/average_v2_diff_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()
    
def new_der_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 3.4
    turbine1 = SuctionSide.functionOfLambda(inputData,('affineangle',), 'fastparam', ('uniform',), 'average', False)
    turbine2 = PressureSide(inputData, 'fastparam', ('uniform',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[3].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[2].label1.set_visible(False)
    yticks[3].label1.set_visible(False)
    yticks[4].label1.set_visible(False)
    yticks[5].label1.set_visible(False)
    yticks[7].label1.set_visible(False)
    yticks[8].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/new_der_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()
    
def good_der_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 3.4
    turbine1 = SuctionSide.functionOfLambda(inputData,('affineangle',), 'param', ('affinechord',), 'average', False)
    turbine2 = PressureSide(inputData, 'param', ('chordlength',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[3].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[2].label1.set_visible(False)
    yticks[4].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/good_der_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()

def good_9_plot(shw = False):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    lambd = 3.4
    turbine1 = SuctionSide.functionOfLambda(inputData,('affineangle',), 'param', ('affinechord',), 'average', True)
    turbine2 = PressureSide(inputData, 'param', ('chordlength',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor = 'white')
    turbine1(lambd).plot2D(100, True, fig)
    turbine2.plot2D(100, True, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    
    ax = plt.gca()
    ax.set_frame_on(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_aspect('equal')
    
    xticks = ax.xaxis.get_major_ticks()
    xticks[1].label1.set_visible(False)
    xticks[2].label1.set_visible(False)
    xticks[3].label1.set_visible(False)
    xticks[4].label1.set_visible(False)
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    yticks[2].label1.set_visible(False)
    yticks[3].label1.set_visible(False)
    yticks[4].label1.set_visible(False)
    yticks[5].label1.set_visible(False)
    yticks[7].label1.set_visible(False)
    yticks[8].label1.set_visible(False)
    
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
    ax.add_artist(Line2D( (xmin, xmin), (ymin, ymax), color='black', linewidth=2))
    
    if not shw:
        plt.savefig('../../latex/img/good_9_curve.pdf', format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()

def pres_curve(shw = False, nrlines = 1):
    inputData = [51.5, 90-15.92, 8., 1.1817, 75.4075, 29.2, 1., .0555, .0220, 30., 2.0]
    turbine1 = SuctionSide.functionOfLambda(inputData,('affineangle',), 'param', ('affinechord',), 'average', False)
    turbine2 = PressureSide(inputData, 'param', ('chordlength',), 'radius')
    leading, trailing = leadingEdge(inputData), trailingEdge(inputData)
    fig = plt.figure(facecolor='white')
    turbine1(3.3).plot2D(100, False, fig, False, 'k-')
    if nrlines >= 2:
        turbine1(3.2).plot2D(100, False, fig, False, 'k-.')
        if nrlines >= 3:
            turbine1(3.4).plot2D(100, False, fig, False, 'k--')
    turbine2.plot2D(100, False, fig, False)
    leading.plot2D(100, False)
    trailing.plot2D(100, False)
    ax = plt.gca()
    ax.axis([-0.01,1.01,-0.1,1.3])
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.set_aspect('equal')
    if not shw:
        plt.savefig('../../presentation/img/diff_blade_curve_{}.pdf'.format(nrlines), format='pdf', dpi = 1200, bbox_inches='tight', pad_inches=0)
    if shw:
        plt.show()


if __name__ == '__main__':
    
    shw = 1
#    geometryplot(shw)
#    arcsplot(shw)
#    diff_blade_plot(shw)
    for i in [1,2,3]:
        pres_curve(0,i)
    """
    simp_plot(shw)
    av_v2_plot(shw)
    av_v2_diff_plot(shw)
    new_der_plot(shw)
    good_der_plot(shw)
    good_9_plot(shw)
    """
#geometryplot()