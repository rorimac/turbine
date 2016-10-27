# -*- coding: utf-8 -*-
from __future__ import division
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import functools
import spline

def basisarray(degree, knots):
    
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
    basislist = [[(lambda x, i = j, deg = degr: basisfunc(x, i, deg)) for j in range(len(knots) - 1 - degr)] for degr in range(degree + 1)]
    return basislist
    
def basisderivative(deriv, basislist, knots):
    degree = len(basislist) - 1
    if type(deriv) != int:
        raise TypeError, "Deriv has to be an integer"
    if deriv < 1:
        raise TypeError, " Deriv has to be 1 or larger"
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

def spacing(points,  domain = (0,1), method = 'uniform', *args):
    
    if type(points) not in [tuple, list, np.ndarray]:
        raise spline.SplineError, "The points must be represented as a tuple, list or numpy array"
    try:
        points = np.array(points)
    except:
        raise spline.SplineError, "The points cannot be converted into a numpy array"
    if len(points) < 2:
        method = 'uniform'
    
    if type(domain) not in [tuple, list, np.ndarray]:
        raise spline.SplineError, "The domain must be represented as a tuple, list or numpy array"
    if len(domain) != 2:
        raise spline.SplineError, "The domain must be a starting point and endingpoint only, for example (0.,1.)"
    if type(domain[0]) not in [int, float] or type(domain[1]) not in [int, float]:
        raise spline.SplineError, "The domain must be an array like object of length 2 and containing only two integer or floats"
    
    if not type(method) == str:
        raise TypeError, "The method name must be a string"
    if not method in ['uniform', 'chordlength', 'centripedal', 'affinechord', 'affineangle']:
        raise spline.SplineError, """method name must be one of the allowed values 
        'uniform', 'chordlength', 'centripedal', 'affinechord' or 'affineangle'"""
    
    if method == 'uniform':
        h_list = uniform_h(points, domain)
    elif method == 'chordlength':
        h_list = centripedal_h(points, domain, 1)
    elif method == 'centripedal':
        if len(args) == 0:
            h_list = centripedal_h(points, domain, 0.5)
        elif len(args) == 1:
            h_list = centripedal_h(points, domain, args[0])
        else:
            raise spline.SplineError, "centripedal method only accepts one alpha value"
    elif method == 'affinechord':
        h_list = affineInvariant_h(points, domain)
    elif method == 'affineangle':
        if len(args) == 0:
            h_list = affineInvariantAngle_h(points, domain, 1.5)
        elif len(args) == 1:
            h_list = affineInvariantAngle_h(points, domain, args[0])
        else:
            raise spline.SplineError, "affineangle method only accepts one delta value"
    else:
        raise TypeError, "method: {} not implemented".format(method)
    
    domainlength = domain[1] - domain[0]
    t_values = np.zeros(len(h_list) + 1)
    t_values[0], t_values[-1] = domain
    if len(t_values) == 2:
        return t_values
    for i in range(1,len(t_values) - 1):
        t_values[i] = t_values[i - 1] + h_list[i - 1]*domainlength
    return t_values

def uniform_h(points, domain):
    
    return np.array([(domain[1] - domain[0])/(len(points) - 1.)]*(len(points) - 1))

def centripedal_h(points, domain, alpha = 0.5):
    
    if not type(alpha) in [int, float]:
        raise TypeError, "alpha value has to be an integer or a float"
    
    lengths = []
    if not len(np.shape(points[0])) in [0,1]:
        raise spline.SplineError, "The individual points has to be vectors or scalars"
    for i in range(1, len(points)):
        if not len(np.shape(points[i])) in [0,1]:
            raise spline.SplineError, "The individual points has to be vectors or scalars"
        lengths.append(np.linalg.norm(points[i] - points[i - 1])**alpha)
    totallength = sum(lengths)
    return lengths/totallength

def affineInvariantMetric_h(points):
    try:
        points = np.array(points)
    except:
        raise TypeError, "points not of a recognized type. Must be array-like"
    if not np.shape(points)[1] == 2:
        raise TypeError, "This method only woks for two-dim vectors arranged as ((x0,y0),(x1,y1),...)"
        
    pointlen = len(points)        
    x_bar = sum(points[:,0])/pointlen; y_bar = sum(points[:,1])/pointlen
    sigma_x = sum([(x - x_bar)**2 for x in points[:,0]])/pointlen
    sigma_y = sum([(y - y_bar)**2 for y in points[:,1]])/pointlen
    sigma_xy = sum([(x - x_bar)*(y - y_bar) for (x,y) in points[:,]])/pointlen
    g = sigma_x*sigma_y - sigma_xy**2
    
    metricmatrix = np.array([[sigma_y/g, -sigma_xy/g],[-sigma_xy/g, sigma_x/g]])
    
    h_list = [np.sqrt(np.dot(np.dot((Xi - Yi),metricmatrix),(Xi - Yi).T)) for (Xi, Yi) in zip(points[:-1,:],points[1:,:])]
    if not len(h_list) + 1 == pointlen:
        raise spline.SplineError, "h_list has not been calculated correctly"
    
    return h_list
    
def affineInvariantMetric(points, Pi, Pj):
    try:
        points = np.array(points)
    except:
        raise TypeError, "points not of a recognized type. Must be array-like"
    if not np.shape(points)[1] == 2:
        raise TypeError, "This method only woks for two-dim vectors arranged as ((x0,y0),(x1,y1),...)"
    
    pointlen = len(points)        
    x_bar = sum(points[:,0])/pointlen; y_bar = sum(points[:,1])/pointlen
    sigma_x = sum([(x - x_bar)**2 for x in points[:,0]])/pointlen
    sigma_y = sum([(y - y_bar)**2 for y in points[:,1]])/pointlen
    sigma_xy = sum([(x - x_bar)*(y - y_bar) for (x,y) in points[:,]])/pointlen
    g = sigma_x*sigma_y - sigma_xy**2
    
    metricmatrix = np.array([[sigma_y/g, -sigma_xy/g],[-sigma_xy/g, sigma_x/g]])
    
    h = np.sqrt(np.dot(np.dot((Pi - Pj),metricmatrix),(Pi - Pj).T))
    
    return h

def affineInvariant_h(points, domain):
    try:
        points = np.array(points)
    except:
        raise TypeError, "points not of a recognized type. Must be array-like"
    if not np.shape(points)[1] == 2:
        raise TypeError, "This method only works for two-dim vectors arranged as ((x0,y0),(x1,y1),...)"
    
    lengths = affineInvariantMetric_h(points)
    totallength = sum(lengths)
    return lengths/totallength

def affineInvariantAngle_h(points, domain, delta = 1.5):
    try:
        points = np.array(points)
    except:
        raise TypeError, "points not of a recognized type. Must be array-like"
    if not np.shape(points)[1] == 2:
        raise TypeError, "This method only works for two-dim vectors arranged as ((x0,y0),(x1,y1),...)"
    
    points = np.vstack([points[0],points,points[-1]]) #This is done so that d_{-1} and d_{n} will equal 0
    dlengths = affineInvariantMetric_h(points)
    
    def theta_i(di1, di2, Pi, Pj):
        if di1 == 0. or di2 == 0.:
            alpha_i = 0
        else:
            alpha_i = np.pi - np.arccos((di1**2 + di2**2 - (affineInvariantMetric(points, Pi, Pj))**2)/(2.*di1*di2))
        return min(np.pi/2, alpha_i)
    
    lengths = []
    for i in range(1, len(dlengths) - 1):
        
        lengths.append(dlengths[i]*(1 + (delta*theta_i(dlengths[i-1],dlengths[i],points[i-1],points[i+1])*dlengths[i-1])/(dlengths[i-1] + dlengths[i])
        + (delta*theta_i(dlengths[i],dlengths[i+1],points[i],points[i+2])*dlengths[i+1])/(dlengths[i] + dlengths[i + 1])))
    
    if not len(lengths) + 3 == len(points):
        raise spline.SplineError, "h_list has not been calculated correctly"
    
    totallength = sum(lengths)
    return lengths/totallength

def knotAveraging(tparam, deg, domain):
    """
    Create the knot vector by averaging the parameter values.
    """
    tlen = len(tparam)
    if tlen <= deg:
        raise TypeError, "Not enough parameter values. Has to be strictly more than the degree"
    
    start = np.ones(deg + 1)*domain[0]
    stop = np.ones(deg + 1)*domain[1]
    length = domain[1] - domain[0]
    
    center = np.empty(tlen - deg - 1)
    for i in range(len(center)):
        s, e = int(i + 1), int(i + 1 + deg)
        center[i] = domain[0] + length/deg*sum(tparam[s:e])
    
    return np.hstack([start, center, stop])
