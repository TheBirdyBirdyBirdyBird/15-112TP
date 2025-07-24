import math
import numpy as np

class Renderer():
    def __init__(self, app):
        self.app = app
        self.triangles = []
        self.cam = [0, 0, 0] # x, y, z 
        self.dir = [0, 0] # only hor + vert
        self.width = 90
        self.height = 90
        self.rays = [100, 100]
        self.gSrcs = []
        self.lSrcs = []

    def render(self):
        view = []
        i = 0
        for y in range(-int(self.height/2), int(self.height/2), int(self.height/self.rays[1])):
            view.append([])
            j = 0
            for x in range(-int(self.width/2), int(self.width/2), int(self.width/self.rays[0])):
                view[-1].append(self.castRay(x, y))
                if i >= 1 and j >= 1:
                    self.renderSquare(view[j-1][i-1], view[j-1][i], view[j][i-1], view[j][i])
                j += 1
            i += 1
    
    def castRay(x, y):
        gX = math.radians(self.dir[0] + x)
        gY = math.radians(self.dir[1] + y) 
        # [math.sin(gX), math.cos(gX)] is length 1. Multiplying math.cos(gY) ensures length of 1 after 
        # addition of z-term
        dirVec = np.array([math.sin(gX) * math.cos(gY), math.cos(gX) * math.cos(gY), math.sin(gY)])
        for triangle in self.triangles:
            if np.cross(triangle.getNormal(), dirVec) == 0:
                continue
            


class Shape():
    def __init__(self, faces):
        self.faces = faces        

class Triangle():
    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
    def getNormal(self):
        return np.cross((self.p2-self.p1), (self.p3-self.p1))


class Vertex():
    def __init__(self, pos):
        self.pos = np.array(pos)
