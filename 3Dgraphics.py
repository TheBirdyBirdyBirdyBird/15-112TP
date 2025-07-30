from cmu_graphics import *
import math, time
import numpy as np
import numba as nb
from PIL import Image

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
    normal = np.cross((p2-p1), (p3-p1))
    normal /= np.linalg.norm(normal)
    return np.array([p1, p2, p3, normal, np.array(color)])
    

def Sphere(origin, radius, color):
    return np.array(origin + [radius] + color)

def Light(pos):
    return np.array(pos)

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
    app.res = [75, 75]
    app.fov = [π/2, π/2]
    app.dir = [0, 0]
    app.pos = np.array([0.0, 0.0, 0.0])
    app.pixelSize = [app.width/app.res[0], app.height/app.res[1]]
    app.angRes = [app.fov[0]/app.res[0], app.fov[1]/app.res[1]]
    tri = Triangle([[-0.1, 1.0, 1.9], [0.1, 1.0, 2.0], [0.0, 1.0, 2.1]], [255.0, 255.0, 255.0])
    tri2 = Triangle([[10.0, 20.0, 5.0], [20.0, 10.0, 5.0], [12.5, 12.5, 20.0]], [255.0, 255.0, 255.0])
    floor = Triangle([[-10.0, 10.0, -1.0], [10.0, 10.0, -1.0], [0.0, -10.0, -1.0]], [122.0, 122.0, 122.0])
    source = Light([10, 10, 10])
    sph1 = Sphere([0.0, 5.0, 2.0], 1.0, [255.0, 255.0, 255.0])
    app.shapes = [np.array([tri, floor, tri2]), np.array([sph1])]
    app.lights = [source]
    app.cursor = [0, 0]
    app.t = time.time()

def redrawAll(app):
    pixels = []
    Ix, Iy, Iz = app.pos[0], app.pos[1], app.pos[2]
    triangles = app.shapes[0]
    spheres = app.shapes[1]
    v = app.dir[1] - app.fov[1]/2
    for z in range(app.res[1]):
        t = app.dir[0] - app.fov[0]/2
        for x in range(app.res[0]):
            dx, dy, dz = np.sin(t)*np.cos(v), np.cos(t)*np.cos(v), np.sin(v)
            pixels.append(tuple(getColor(Ix, Iy, Iz, dx, dy, dz, triangles, spheres, app.lights).astype(int)))
            t += app.angRes[0]
        v += app.angRes[1]
    frame = Image.new(mode="RGB", size=app.res)
    frame.putdata(pixels)
    frame = frame.transpose(Image.FLIP_TOP_BOTTOM)
    drawImage(CMUImage(frame), 0, 0, width = app.width, height = app.height)

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
    processKey(app, key, modifiers)
def onKeyHold(app, keys, modifiers):
    for key in keys:
        processKey(app, key, modifiers)

def processKey(app, key, modifiers):
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
    triangle = np.ascontiguousarray(triangle)
    dir = np.ascontiguousarray(dir)
    if np.dot(triangle[3], dir) == 0:
        return False
    barycentric = make3x3NpMatrix(-dir, (triangle[1]-triangle[0]), (triangle[2]-triangle[0]))
    dist, u, v = np.linalg.solve(barycentric, pos-triangle[0])
    if (dist > 0 and u > 0 and v > 0 and 1-u-v > 0):
        return dist
    else:
        return False
    
@nb.jit(nb.f8(nb.f8[:], nb.f8[:], nb.f8[:]))
def intersectSphere(dir, pos, sphere):
    dir = np.ascontiguousarray(dir)
    relPos = sphere[:3] - pos
    projection = np.dot(dir, relPos)
    if projection < 0: return False
    d = (np.linalg.norm(relPos) ** 2 - projection ** 2) ** 0.5
    if d > sphere[3]: return False
    dist = projection - (sphere[3] ** 2 - d ** 2) ** 0.5
    return dist

def findIntersection(dir, pos, triangles, spheres):
    minDist = None
    color = np.array([0.0, 0.0, 0.0])
    for triangle in triangles:
        intersect = intersectTriangle(dir, pos, triangle)
        if intersect:
            if minDist == None or intersect < minDist:
                minDist = intersect
                color = triangle[4]
                normal = triangle[3]
    for sphere in spheres:
        intersect = intersectSphere(dir, pos, sphere)
        if intersect:
            if minDist == None or intersect < minDist:
                minDist = intersect
                color = sphere[4:]
                normal = (pos + dir * intersect - sphere[:3]) / sphere[3]
    if minDist != None:
        return np.array(np.concatenate((np.array([minDist]), color, normal)))
    else: return None

#@nb.jit(nb.f8[:](nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8, nb.f8[:,:,:], nb.f8[:,:,:]))
def getColor(x, y, z, dx, dy, dz, triangles, spheres, lights):
    #background = [0, 0, 0] #rgb(abs(dx)*255, abs(dy)*255, abs(dz)*255)
    minDist = None
    color = np.array([0.0, 0.0, 0.0])
    dir = np.array([dx, dy, dz])
    pos = np.array([x, y, z])
    intersection = findIntersection(dir, pos, triangles, spheres)
    if isinstance(intersection, type(None)):
        return color
    rayPos = pos + dir * (intersection[0] - 0.001)
    illumination = 0
    for light in lights:
        rayDir = light - rayPos
        dist = np.linalg.norm(rayDir)
        rayDir /= dist
        blockers = findIntersection(rayDir, rayPos, triangles, spheres)
        if isinstance(blockers, type(None)) or blockers[0] > dist:
            illumination += 100 * abs(np.dot(intersection[4:], rayDir)) * 1/dist ** 2
    color = intersection[1:4] * illumination + 25
    return color

main()
