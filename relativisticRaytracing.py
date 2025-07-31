# 3rd overhaul

from cmu_graphics import *
import math, time
import numpy as np
import numba as nb
from PIL import Image
import colorsys

# apparently this works
# global π
# π = 3.14159

# bound x between low and high
def constrain(x, low, high):
    return max(min(x, high), low)

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

class Triangle():
    def __init__(self, vertices, color):
        self.p1 = np.array(vertices[0])
        self.p2 = np.array(vertices[1])
        self.p3 = np.array(vertices[2])
        self.normal = np.cross((self.p2-self.p1), (self.p3-self.p1))
        self.normal /= np.linalg.norm(self.normal)
        self.color = color

    def serialize(self): # converts to a list for easy usage with numba
        return np.array([self.p1, self.p2, self.p3, self.normal, np.array(self.color)])
    
class Sphere():
    def __init__(self, origin, radius, color):
        self.origin = origin
        self.radius = radius
        self.color = color

    def serialize(self):
        return np.array(self.origin + [self.radius] + self.color)

class Light():
    def __init__(self, origin, intensity, color):
        self.origin = origin
        self.intensity = intensity
        self.color = color

    def serialize(self):
        return np.array(self.origin + [self.intensity] + self.color)

def onAppStart(app):
    app.background = 'black'
    app.res = [75, 75] # resolution
    app.fov = [math.pi/2, math.pi/2] # field of view
    app.dir = [0, 0]
    app.pos = np.array([11.0, -15.0, 11.0])
    app.pixelSize = [app.width/app.res[0], app.height/app.res[1]]
    app.angRes = [app.fov[0]/app.res[0], app.fov[1]/app.res[1]] # angular resolution
    # some showcase objects
    tri = Triangle([[-0.1, 1.0, 1.9], [0.1, 1.0, 2.0], [0.0, 1.0, 2.1]], [255.0, 255.0, 255.0])
    tri2 = Triangle([[10.0, 20.0, 5.0], [20.0, 10.0, 5.0], [12.5, 12.5, 20.0]], [255.0, 255.0, 255.0])
    floor = Triangle([[-10.0, 10.0, -1.0], [10.0, 10.0, -1.0], [0.0, -10.0, -1.0]], [122.0, 122.0, 122.0])
    source = Light([10, 10, 10], 1, [247, 245, 207])
    sph1 = Sphere([0.0, 5.0, 2.0], 1.0, [255.0, 122.0, 0.0])
    app.shapes = [np.array([tri, floor, tri2]), np.array([sph1])]
    app.lights = [source]

    app.cursor = [0, 0]
    app.t = time.time() # track delay between frames
    app.lightSpeed = 300000000
    app.motion = [0, 0, 0]
    app.enableRelativity = True

def onStep(app):
    print(time.time()-app.t) # delay between frames
    app.t=time.time()
    print(app.pos, app.dir)

def onMousePress(app, mouseX, mouseY):
    app.cursor = [mouseX, mouseY]

def onMouseDrag(app, mouseX, mouseY):
    # check movement of cursor to adjust camera direction
    dx = mouseX - app.cursor[0]
    dz = mouseY - app.cursor[1]
    app.cursor = [mouseX, mouseY]
    app.dir[0] += dx * 0.01
    app.dir[1] -= dz * 0.01

def onKeyPress(app, key, modifiers):
    if key == 'p': # putting this inside onKeyHold caused it to switch randomly unless I flicked the button very fast
        app.motion = [0,0,0]
        app.enableRelativity = not app.enableRelativity
def onKeyHold(app, keys, modifiers):
    processKey(app, keys[0], modifiers)

def processKey(app, key, modifiers):
    key = key.lower()
    if key == 'w' or key == 'up':
        dx = math.sin(app.dir[0]) * 0.1
        dy = math.cos(app.dir[0]) * 0.1
        app.pos[0] += dx
        app.pos[1] += dy
        app.motion = [dx * 10, dy * 10, 0]
    elif key == 'a' or key == 'left':
        dy = math.sin(app.dir[0]) * 0.1
        dx = math.cos(app.dir[0]) * 0.1
        app.pos[0] -= dx
        app.pos[1] += dy
        app.motion = [-dx * 10, dy * 10, 0]
    elif key == 's' or key == 'down':
        dx = math.sin(app.dir[0]) * 0.1
        dy = math.cos(app.dir[0]) * 0.1
        app.pos[0] -= dx
        app.pos[1] -= dy
        app.motion = [-dx * 10, -dy * 10, 0]
    elif key == 'd' or key == 'right':
        dy = math.sin(app.dir[0]) * 0.1
        dx = math.cos(app.dir[0]) * 0.1
        app.pos[0] += dx
        app.pos[1] -= dy
        app.motion = [dx * 10, -dy * 10, 0]
    elif key == 'z' or key == 'space':
        app.pos[2] += 0.1
        app.motion = [0, 0, 1]
    elif key == 'x':
        app.pos[2] -= 0.1
        app.motion = [0, 0, -1]
    #adjust speed of light based on log scale. shift for higher sensitivity
    elif key == 'e':
        logMetric = math.log(app.lightSpeed) * 20.5
        if 'shift' in modifiers:
            if logMetric >= 1:
                app.lightSpeed = math.e ** ((logMetric - 1)/20.5)
        elif logMetric >= 10:
            app.lightSpeed = math.e ** ((logMetric - 10)/20.5)
    elif key == 'r':
        logMetric = math.log(app.lightSpeed) * 20.5
        if 'shift' in modifiers:
            if logMetric <= 399.2:
                app.lightSpeed = math.e ** ((logMetric + 1)/20.5)
        elif logMetric <= 391:
            app.lightSpeed = math.e ** ((logMetric + 10)/20.5)

    #elif key == 'i':
        #adjustVariable(app)

# intended to let user set certain parameters or add new objects. However, the input() function interacts 
# strangely with cmu_graphics and it oftentimes causes the program to crash or freeze permanently. 
def adjustVariable(app):
    query = int(input("adjust variable: 1=x, 2=y, 3=z, 4=speed of light, 5=3d element, 6=cancel\n"))
    if query == 6: 
        print('cancel')
        return None
    elif query <= 4 and query >= 1:
        value = float(input("enter new value\n"))
        if query == 1:
            app.pos[0] = value
        elif query == 2:
            app.pos[1] = value
        elif query == 3:
            app.pos[2] = value
        elif query == 4:
            print(app.lightSpeed)
            app.lightSpeed = value
            print(app.lightSpeed)
    elif query == 5:
        shapeType = int(input("1=Triangle, 2=Sphere, 3=Light\n"))
        if shapeType == 1:
            parameters = input("12 space separated numbers: x1 y1 z1 x2 y2 z2 x3 y3 z3 red green blue\n")
            parameters = [int(parameter) for parameter in parameters.split()]
            triangle = Triangle(parameters[:9], parameters[9:])
            app.shapes[0].append(triangle)
        elif shapeType == 2:
            parameters = input("7 space separated numbers: x y z radius red green blue\n")
            parameters = [int(parameter) for parameter in parameters.split()]
            sphere = Sphere(parameters[:3], parameters[3], parameters[4:])
            app.shapes[1].append(sphere)
        elif shapeType == 3:
            parameters = input("7 space separate numbers: x y z intensity red green blue\n")
            parameters = [int(parameter) for parameter in parameters.split()]
            light = Light(parameters[:3], parameters[3], parameters[4:])
            app.lights.append(light)

def redrawAll(app):
    renderScreen(app)
    if app.enableRelativity:
        drawLightSlider(app)

def renderScreen(app):
    pixels = []
    Cx, Cy, Cz = app.pos[0], app.pos[1], app.pos[2]
    triangles = [triangle.serialize() for triangle in app.shapes[0]]
    spheres = [sphere.serialize() for sphere in app.shapes[1]]
    lights = [light.serialize() for light in app.lights]
    v = app.dir[1] - app.fov[1]/2
    for z in range(app.res[1]):
        t = app.dir[0] - app.fov[0]/2
        for x in range(app.res[0]):
            dx, dy, dz = np.sin(t)*np.cos(v), np.cos(t)*np.cos(v), np.sin(v)
            color = getColor( Cx, Cy, Cz, dx, dy, dz, triangles, spheres, lights).astype(int)
            if app.enableRelativity:
                color = applyRelativity(color, app.motion, [dx, dy, dz], app.lightSpeed)
            pixels.append(tuple(color))
            t += app.angRes[0]
        v += app.angRes[1]
    frame = Image.new(mode="RGB", size=app.res)
    frame.putdata(pixels)
    frame = frame.transpose(Image.FLIP_TOP_BOTTOM)
    drawImage(CMUImage(frame), 0, 0, width = app.width, height = app.height)

def applyRelativity(color, motion, rayDir, c):
    relSpeed = -np.dot(motion, rayDir)
    wavelengthRatio = ((1 + relSpeed/c) / (1 - relSpeed/c)) ** 0.5 # Doppler factor, Wikipedia
    color = list(colorsys.rgb_to_hsv(*(color/255))) # HSV colors are easier to adjust to match relativistic effects

    # Doppler effect
    # convert hue to wavelength linearly, red (0)->700, deep purple (5/6)->380
    wavelength = (1-color[0]) * 384 + 316 
    
    observed = wavelengthRatio * wavelength # Wikipedia
    if observed > 700:
        color[2] /= observed - 699
        observed = 700
    elif observed < 380:
        color[2] /= 381 - observed
        observed = 380
    color[0] = (316 - observed) / 384 + 1

    # Headlight effect
    color[1] = constrain(color[1] * wavelengthRatio ** 2, 0, 1) # Wikipedia

    #reformat data
    color = (np.array(colorsys.hsv_to_rgb(*color)) * 255).astype(int)
    color[0] = constrain(color[0], 0, 255)
    color[1] = constrain(color[1], 0, 255)
    color[2] = constrain(color[2], 0, 255)
    return color

def drawLightSlider(app):
    drawRect(50, 50, 400, 20, fill='white', border='black')
    sliderPos = math.log(app.lightSpeed) * 20.5 + 50 # uses logarithmic scale
    drawRect(sliderPos, 45, 30, 30, fill='black', border='white')
    drawLabel(f'light speed is {rounded(app.lightSpeed)} m/s', 250, 90, fill='blue', size=20)

@nb.jit(nb.f8(nb.f8[:], nb.f8[:], nb.f8[:,:]))
def intersectTriangle(dir, pos, triangle):
    # Moller-Trumbore algorithm, did not use code references to help. 
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
    dir = np.ascontiguousarray(dir) # dot product is faster on contigious arrays
    relPos = sphere[:3] - pos
    projection = np.dot(dir, relPos)
    if projection < 0: return False
    altitude = (np.linalg.norm(relPos) ** 2 - projection ** 2) ** 0.5 # I can't think of a good name
    if altitude > sphere[3]: return False
    dist = projection - (sphere[3] ** 2 - altitude ** 2) ** 0.5
    return dist

def findIntersection(dir, pos, triangles, spheres):
    # loop through all shapes for closest intersection
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

def getColor(x, y, z, dx, dy, dz, triangles, spheres, lights):
    color = np.array([0.0, 0.0, 0.0])
    dir = np.array([dx, dy, dz])
    pos = np.array([x, y, z])
    intersection = findIntersection(dir, pos, triangles, spheres) #first find what the ray hits
    if isinstance(intersection, type(None)):
        return color # return default color if it hits nothing
    rayPos = pos + dir * (intersection[0] - 0.001) # slightly behind where the ray intersects, prepares for lighting
    illumination = np.array([0.1, 0.1, 0.1]) # start with some subtle ambient lighting
    for light in lights: # if light has line of sight with intersection location, apply lighting from that light
        rayDir = light[:3] - rayPos 
        dist = np.linalg.norm(rayDir)
        rayDir /= dist
        blockers = findIntersection(rayDir, rayPos, triangles, spheres) # check line of sight
        if isinstance(blockers, type(None)) or blockers[0] > dist:
            illumination += light[4:] * light[3] * abs(np.dot(intersection[4:], rayDir)) * 1/dist ** 2 # apply
    color = np.multiply(intersection[1:4], illumination)
    return color

def main():
    runApp(width=500, height=500)

main()
