from cmu_graphics import *
import cv2
import numpy as np

def onAppStart(app):
    app.x = 0
    app.stepsPerSecond = 100
    app.offset = 0
    app.url = 'smalltest.png'#'Mandelbrot-0.7003269770890052_0.3350680938219896_-0.6832962467853403_0.3192798440209425_5000_1.png'
    app.img = cv2.imread(app.url)
    app.pts_src = np.array([[0, 0], [0, 100], [100, 0],[100, 100]])


def redrawAll(app):
    
    drawImage('slant.png', 0, 0, width=256, height=256)
    drawRect(0, 0, 100, 100, fill=rgb(app.offset*2, 0, 0))

def onStep(app):
    app.offset += 0.1
    print(app.offset)
    pts_dst = np.array([[0, 0], [app.offset, 100], [100, 0],[100+app.offset, 100]])
    h, status = cv2.findHomography(app.pts_src, pts_dst)
    im_out = cv2.warpPerspective(app.img, h, (app.img.shape[1], app.img.shape[0]))
    im_out = cv2.resize(im_out, (64, 64))
    cv2.imwrite('slant.png', im_out)    #print(app.x)
    #print('test')

def main():
    runApp(width=1000, height=1000)
main()
