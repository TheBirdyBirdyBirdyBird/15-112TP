from cmu_graphics import *
import math, time
import numpy as np
import numba as nb

global π
π = 3.14159

'''
clockwise
       y
  #####|#####
  ####/^\####
-x-----+-----x
  #####|#####
  #####|#####
      -y
'''

def types(data):
    for thing in data:
        print('type: ', nb.typeof(thing), thing)

def Triangle(vertices, color):
    p1 = np.array(vertices[0])
    p2 = np.array(vertices[1])
    p3 = np.array(vertices[2])
    return np.array([p1, p2, p3, np.cross((p2-p1), (p3-p1)), np.array(color)])

def Sphere(origin):
    pass

# using numba is 6% faster
@nb.jit(nb.f8[:,:](nb.i4, nb.i4))
def numba_zeros(x, y):
    return np.zeros((x, y))

# converts cols into matrix, also tells numba the correct types
@nb.jit(nb.f8[:,:](nb.f8[:], nb.f8[:], nb.f8[:]))
def make3x3NpMatrix(a1, a2, a3):
    matrix = numba_zeros(3, 3)
    matrix[0] = a1
    matrix[1] = a2
    matrix[2] = a3
    return matrix.transpose()

def onAppStart(app):
    app.background = 'black'
    app.setMaxShapeCount(100000)
    app.res = [100, 100]
    app.fov = [π/2, π/2]
    app.dir = [0, 0]
    app.pos = np.array([0.0, 0.0, 0.0])
    app.pixelSize = [app.width/app.res[0], app.height/app.res[1]]
    app.angRes = [app.fov[0]/app.res[0], app.fov[1]/app.res[1]]
    tri = Triangle([[-0.1, 1.0, -0.1], [0.1, 1.0, 0.0], [0.0, 1.0, 0.1]], [255.0, 255.0, 255.0])
    floor = Triangle([[-10.0, 10.0, -1.0], [10.0, 10.0, -1.0], [0.0, -10.0, -1.0]], [122.0, 122.0, 122.0])
    app.shapes = [np.array([tri, floor]), np.array([tri])]
    app.cursor = [0, 0]
    app.t = time.time()

def redrawAll(app):
    pixels = []
    Ix, Iy, Iz = app.pos[0], app.pos[1], app.pos[2]
    triangles = app.shapes[0]
    spheres = app.shapes[1]
    v = app.dir[1] - app.fov[1]/2
    rects = 0
    for z in range(app.res[1]):
        t = app.dir[0] - app.fov[0]/2
        for x in range(app.res[0]):
            dx, dy, dz = np.sin(t)*np.cos(v), np.cos(t)*np.cos(v), np.sin(v)
            color = getColor(Ix, Iy, Iz, dx, dy, dz, triangles, spheres)
            if color.all() != 0.0:
                rects += 1
                drawRect(x * app.pixelSize[0], app.height - z * app.pixelSize[1], app.pixelSize[0], app.pixelSize[1], fill=rgb(*color), align='bottom-left')
            t += app.angRes[0]
        v += app.angRes[1]
    print(rects)

def onStep(app):
    print(time.time()-app.t)
    app.t=time.time()

def onMousePress(app, mouseX, mouseY):
    app.cursor = [mouseX, mouseY]

def onMouseDrag(app, mouseX, mouseY):
    dx = mouseX - app.cursor[0]
    dz = mouseY - app.cursor[1]
    app.cursor = [mouseX, mouseY]
    app.dir[0] += dx * 0.01
    app.dir[1] -= dz * 0.01

def onKeyPress(app, key, modifiers):
    if 'shift' in modifiers:
        app.pos[2] -= 0.1
    elif key == 'w' or key == 'up':
        dx = math.sin(app.dir[0]) * 0.1
        dy = math.cos(app.dir[0]) * 0.1
        app.pos[0] += dx
        app.pos[1] += dy
    elif key == 'a' or key == 'left':
        dy = math.sin(app.dir[0]) * 0.1
        dx = math.cos(app.dir[0]) * 0.1
        app.pos[0] -= dx
        app.pos[1] += dy
    elif key == 's' or key == 'down':
        dx = math.sin(app.dir[0]) * 0.1
        dy = math.cos(app.dir[0]) * 0.1
        app.pos[0] -= dx
        app.pos[1] -= dy
    elif key == 'd' or key == 'right':
        dy = math.sin(app.dir[0]) * 0.1
        dx = math.cos(app.dir[0]) * 0.1
        app.pos[0] += dx
        app.pos[1] -= dy
    elif key == 'space':
        app.pos[2] += 0.1
        

def main():
    runApp(width=500, height=500)

@nb.jit(nb.f8(nb.f8[:], nb.f8[:], nb.f8[:,:]))
def intersectTriangle(dir, pos, triangle):
    if np.dot(triangle[3], dir) == 0:
        return False
    barycentric = make3x3NpMatrix(-dir, (triangle[1]-triangle[0]), (triangle[2]-triangle[0]))
    dist, u, v = np.linalg.solve(barycentric, pos-triangle[0])
    if (dist > 0 and u > 0 and v > 0 and 1-u-v > 0):
        return dist
    else:
        return False

'''#@nb.jit(nb.f8[:](nb.f8[:], nb.f8[:], nb.f8[:,:,:], nb.f8[:,:,:]))
def getColor(dir, pos, triangles, spheres):
    #background = [0, 0, 0] #rgb(abs(dx)*255, abs(dy)*255, abs(dz)*255)
    minDist = None
    color = np.array([0.0, 0.0, 0.0])
    for triangle in triangles:
        intersect = intersectTriangle(dir, pos, triangle)
        if intersect:
            if minDist == None or intersect < minDist:
                minDist = intersect
                color = triangle[4]
    return np.array([0.0, 0.0, 0.0])'''


#@nb.jit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:,:,:], nb.f8[:,:,:]))
def getColor(x, y, z, dx, dy, dz, triangles, spheres):
    #background = [0, 0, 0] #rgb(abs(dx)*255, abs(dy)*255, abs(dz)*255)
    minDist = None
    color = np.array([0.0, 0.0, 0.0])
    dir = np.array([dx, dy, dz])
    pos = np.array([x, y, z])
    for triangle in triangles:
        intersect = intersectTriangle(dir, pos, triangle)
        if intersect:
            if minDist == None or intersect < minDist:
                minDist = intersect
                color = triangle[4]
    return color
    


main()
