# -*- coding: utf-8 -*-
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from Nurbs import Crv, Util
from Nurbs.Util import NURBSError
import util as util
from math import radians, pi

class SplineError(NURBSError):
    pass

class Spline(Crv.Crv):
    
    def basisarray(self):
        degree = self.degree
        knots = self.uknots
        
        def knotfraction(u0, u1, u2, u3):
            if u2 == u3: return 0
            return (u0 - u1)/(u2 - u3)
        
        def basisfunc(x, i, deg):
            if deg == 0:
                if knots[i + 1] == knots[-1]:
                    return float(knots[i] <= x <= knots[i + 1])
                if knots[i] == knots[i + 1]: return 0.
                return float(knots[i] <= x < knots[i + 1])
            return knotfraction(x,knots[i], knots[i + deg], knots[i])*basisfunc(
            x, i, deg - 1) + knotfraction(knots[i + deg + 1], x, knots[i + deg + 1], knots[i + 1]
            )*basisfunc(x, i + 1, deg - 1)
        self.basislist = [[(lambda x, i = j, deg = degr: basisfunc(x, i, deg)) for j in range(len(knots) - 1 - degr)] for degr in range(degree + 1)]
        return self.basislist
        
    def basisderivative(self, deriv):
        basislist = self.basislist
        knots = self.uknots
        degree = len(basislist) - 1
        if type(deriv) != int:
            raise SplineError, "Deriv has to be an integer"
        if deriv < 1:
            raise SplineError, " Deriv has to be 1 or larger"
        basisfuncs = basislist[-1 - deriv]
        
        def knotfraction(deg, u1, u2):
            if u1 == u2: return 0
            return 1/float(u1 - u2)
        
        def derivative(x, i, deg, der):
            
            if der == 1:
                return  knotfraction(deg, knots[i + deg], knots[i])*basisfuncs[i](x
                ) - knotfraction(deg, knots[i + deg + 1], knots[i + 1])*basisfuncs[i + 1](x)
            return knotfraction(deg, knots[deg + i], knots[i])*derivative(x, i, deg - 1, der - 1
            ) - knotfraction(deg, knots[deg + i + 1], knots[i + 1])*derivative(
            x, i + 1, deg - 1, der - 1)
        
        derivativelist = [(lambda x, i = j, deg = degree, der = deriv: derivative(x, i, deg, deriv)) for j in range(len(basislist[-1]))]
        return derivativelist
    
    def plot2D(self, n = 25, points = True, fig = None, text = True, color = 'k'):

        pnts = self.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
        knot = self.pnt3D(self.uknots)
        ctrl = self.cntrl[:3]/self.cntrl[3]
        """
        plt.plot(pnts[0], pnts[1], label='parametric bspline')
        if points:
            plt.plot(ctrl[0], ctrl[1], 'ro-', label='control pts', linewidth=.5)
            plt.plot(knot[0], knot[1], 'y+', markersize = 10, markeredgewidth=1.8, label="knots")
        """
        if fig == None:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        if points:
            if text:
                ax.set_title( "B-spline curve, degree={0}".format(self.degree) )
                ax.plot(pnts[0], pnts[1], color, label='parametric bspline')
                ax.plot(ctrl[0], ctrl[1], 'ro-', label='control pts', linewidth=.5)
                ax.plot(knot[0], knot[1], 'y+', markersize = 10, markeredgewidth=1.8, label="knots")
            else:
                ax.plot(pnts[0], pnts[1], color)
                ax.plot(ctrl[0], ctrl[1], 'ro-')
                ax.plot(knot[0], knot[1], 'y+')
        else:
            if text:
                ax.set_title( "B-spline curve, degree={0}".format(self.degree) )
                ax.plot(pnts[0], pnts[1], color, label='parametric bspline')
            else:
                ax.plot(pnts[0], pnts[1], color)

        ax.legend(fontsize='x-small',bbox_to_anchor=(0.95, .9), loc=2, borderaxespad=-1.)
        
    
    def _derivcntrl(self, update = None):
        """
        A support function that calculates all the control points for the derivatives
        """
        
        derivmatrix = [self.cntrl]
        for i in range(1, self.degree):
            cntrlmatrix = np.ones((4, len(derivmatrix[i - 1][0]) - 1))
            for j in range(len(cntrlmatrix[0])):
                for k in range(3):
                    cntrlmatrix[k, j] = (self.degree + 1 - i)*(derivmatrix[i - 1][k, j + 1]-derivmatrix[i - 1][k, j])/(self.uknots[self.degree + i]- self.uknots[i])
            derivmatrix.append(cntrlmatrix.copy())
        return derivmatrix
    
    def nderiv(self, n):
        """
        Computes the n'th derivative of the b-spline and returns it as a b-spline
        as described by De Boor, C. 'On Calculating With B-splines' 1972.
        It is assumed that the curve is clamped to the interval [0,1] with a multiplicity
        one greater than the degree.
        """

        if n >= self.degree:
            raise Util.NURBSError, 'Unable to differentiate up to or more than the degree.'
#        cntrl = self.cntrl
#        knots = self.uknots
        self._dcntrl = self._derivcntrl()
        return Spline(self._dcntrl[n],self.uknots[n:-n])

class Arc(Crv.Arc):
    
    def plot2D(self, n = 25, points = True, color = 'k'):

        pnts = self.pnt3D(np.arange(n + 1, dtype = np.float64)/n)
        knot = self.pnt3D(self.uknots)
        ctrl = self.cntrl[:3]/self.cntrl[3]
        plt.plot(pnts[0], pnts[1], color, label='parametric bspline')
        if points:
            plt.plot(ctrl[0], ctrl[1], 'ro-', label='control pts', linewidth=.5)
            plt.plot(knot[0], knot[1], 'y+', markersize = 10, markeredgewidth=1.8, label="knots")
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.set_title( "b-spline Curve, degree={0}".format(self.degree) )
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.plot(pnts[0], pnts[1], label='parametric bspline')
        ax.plot(ctrl[0], ctrl[1], 'ro-', label='control pts', linewidth=.5)
        ax.plot(knot[0], knot[1], 'y+', markersize = 10, markeredgewidth=1.8, label="knots")
        """