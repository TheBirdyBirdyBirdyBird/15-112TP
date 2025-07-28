from cmu_graphics import *
import cv2
import math
import numpy as np
import threading
import multiprocess
import multiprocessing
import dill

global π
π = 3.14159#265358979

# Möller-Trumbore algorithm to determine intersection with triangles
def intersection(dir, pos, triangle):
    # if triangle is parallel to ray, it does not intersect
    epsilon = 0.001
    if np.dot(triangle.normal, dir) == 0:
        return 0
    # [-dirVec, (p2-p1), (p3-p1)] * [t, u, v] = origin - p1
    # use Cramer's rule to calculate t, u, v
    p1, p2, p3 = triangle.p1, triangle.p2, triangle.p3
    detA = np.linalg.det([-dir, p2-p1, p3-p1])
    dist = np.linalg.det([pos-p1, p2-p1, p3-p1])/detA
    u = np.linalg.det([-dir, pos-p1, p3-p1])/detA
    v = np.linalg.det([-dir, p2-p1, pos-p1])/detA
    # dist < 0 means triangle is behind ray. if u, v, w=1-u-v are <0, there is no intersection
    if (dist > 0 and u > 0 and v > 0 and 1-u-v > 0):
        return dist
    else:
        return 0
    
def radians(deg):
    return deg * π / 180

def degrees(rad):
    return rad * 180 / π

def fRange(start, stop, step):
    while start < stop:
        yield start
        start += step

def constrain(x, low, high):
    return max(min(x, high), low)

class Renderer():
    def __init__(self, app):
        self.app = app
        self.triangles = [Triangle([-0.1, 1, -0.1], [0.1, 1, 0], [0, 1, 0.1])]
        self.pos = np.array([0, 0, 0]) # x, y, z, meters
        self.dir = [0, 0] # only hor + vert, degrees
        self.width = 90 # degrees
        self.height = 90 # degrees
        self.rays = [100, 100]
        self.angRes = [self.width/self.rays[0], self.height/self.rays[1]]
        self.gSrcs = []
        self.lSrcs = []
        self.lumRef = 1
        self.camRad = 0.01 # physical radius of camera in m. Only used for brightness calculations
        self.pixelDims = [app.width/self.rays[0], app.height/self.rays[1]]

    def render(self, batch):
        view = []
        start = self.rays[1]/8*batch
        stop = start + self.rays[1]/8
        for i in range(math.floor(start), math.floor(stop)):
            view.append([])
            y = i * self.angRes[1] - self.height/2
            for j in range(self.rays[0]):
                x = j * self.angRes[0] - self.width/2
                view[-1].append(self.castRay(x, y))
        return view
    
    def renderSquare(self, x, y, p1, p2, p3, p4, app):
        #print(p1, p2, p3, p4)
        # v1: render average of points, assume light source at camera, 100% reflectivity
        # reflect is proportion of total light scattered by the plane at p1
        scatter = (radians(self.angRes[0]/2) * radians(self.angRes[1])/2) / (4 * π)
        absorb = 0 # absorb is proportion of scattered light absorbed by camera
        for dist in [p1, p2, p3, p4]:
            if dist:
                absorb += scatter * math.sin(math.atan(self.camRad / dist) / 2) ** 2
        if absorb == 0:
            return 0
        appLum = math.log(absorb * self.lumRef, 1.1) + 370 # logarithmic brightness is more versatile
        color = rgb(constrain(appLum, 0, 255), constrain(appLum, 0, 255), constrain(appLum, 0, 255))
        #print(appLum)
        drawRect(self.pixelDims[0]*x, self.pixelDims[1]*y, self.pixelDims[0], self.pixelDims[1], 
                 fill=color, align='bottom-right')
        return appLum
        
    def castRay(self, x, y):
        gX = math.radians(self.dir[0] + x)
        gY = math.radians(self.dir[1] + y) 
        # [math.sin(gX), math.cos(gX)] is length 1. Multiplying math.cos(gY) ensures length of 1 after 
        # addition of z-term
        dirVec = np.array([math.sin(gX) * math.cos(gY), math.cos(gX) * math.cos(gY), math.sin(gY)])
        #if (x, y) == (0, 0):
            #print(dirVec)
        minDist = 0 #intersection(dirVec, self.pos, self.triangles[0])
        if len(self.triangles) > 0:
            minDist = intersection(dirVec, self.pos, self.triangles[0])
        for triangle in self.triangles[1:]:
            dist = intersection(dirVec, self.pos, triangle)
            if dist and dist < minDist: # if the intersection passes and is closest yet found
                minDist = dist # update closest intersection
        return minDist

class Shape():
    def __init__(self, faces):
        self.faces = faces        

class Triangle():
    def __init__(self, p1, p2, p3):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
        self.normal = np.cross((self.p2-self.p1), (self.p3-self.p1))
        self.a=float(self.normal[0])
        self.b=float(self.normal[1])
        self.c=float(self.normal[2])
    def isParallel(self, dir, epsilon):
        if abs(dir[0] * self.a + dir[1] * self.b + dir[2] * self.c) < epsilon:
            return True
        else:
            return False


class Vertex():
    def __init__(self, pos):
        self.pos = np.array(pos)

def onAppStart(app):
    app.r = Renderer(app)
    app.setMaxShapeCount(100000)

def redrawAll(app):
    view = []
    for i in range(8):
        for row in app.r.render(i):
            view.append(row)
    
    #p = multiprocessing.Pool(processes=8)
    '''processes = []
    for i in range(8):
        #processes.append(threading.Thread(target=app.r.render, args=(app, i, views)))
        processes.append(multiprocess.Process(target=app.r.render, args=(app, i, views)))
    for process in processes:
        process.start()
    for process in processes:
        process.join()'''
    
    #for row in view: print(row)
    for j in range(len(view)):
        for i in range(len(view[0])):
            if i >= 1 and j >= 1:
                app.r.renderSquare(j, i, view[j-1][i-1], view[j-1][i], view[j][i-1], view[j][i], app)
            
    '''threads = [threading.Thread(target=app.r.render, args=(app, i)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()'''
    print('done')
    #app.r.render(app, 0)

def onStep(app):
    pass

def main():
    runApp(width=1000, height=1000)

main()
