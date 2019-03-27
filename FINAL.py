import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy as sp
from scipy import interpolate,ndimage,signal
import numpy as np
import cv2
import math
import glob
import os
from operator import itemgetter
import shutil
# from skimage import color

def normalize( matrix ):
    abs_sum = np.sum( [ abs( x ) for x in matrix ] )
    return matrix / abs_sum if abs_sum != 0 else matrix

def rotate(arrayx,arrayy,angle):
    array = []
    angle = -angle
    cos = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    for i in range(len(arrayx)):
        x = arrayx[i]*cos - arrayy[i]*sin
        y = arrayx[i]*sin + arrayy[i]*cos
        array.append((x,y))
    return array

# templatexy = [[812.40322581, 661.82258065], [825.9516129, 652.14516129], [839.5, 642.46774194], [851.11290323, 636.66129032], [862.72580645, 630.85483871], [874.33870968, 626.98387097], [885.9516129, 623.11290323], [896.22580645, 626.80645161], [906.5, 630.5], [915.58064516, 626.80645161], [924.66129032, 623.11290323], [940.14516129, 626.98387097], [955.62903226, 630.85483871], [967.24193548, 636.66129032], [978.85483871, 642.46774194], [992.40322581, 654.08064516], [1005.9516129, 665.69354839], [1000.14516129, 679.24193548], [994.33870968, 692.79032258], [986.59677419, 704.40322581], [978.85483871, 716.01612903], [965.30645161, 719.88709677], [951.75806452, 723.75806452], [934.33870968, 731.5], [916.91935484, 739.24193548], [899.5, 735.37096774], [882.08064516, 731.5], [868.53225806, 723.75806452], [854.98387097, 716.01612903], [845.30645161, 710.20967742], [835.62903226, 704.40322581], [827.88709677, 694.72580645], [820.14516129, 685.0483871]]
def rtpairs(r, n):
    for i in range(len(r)):
       for j in range(n[i]):    
        yield r[i], j*(2 * np.pi / n[i])

def scaleTransformTemplate(c, scale, transform):
    for i in range(len(c)):
        c[i] = (c[i] * scale) + transform
    return c

def getStartTemplate():
    T = [33]
    R = [250]
    x = []
    y = []
    for r, t in rtpairs(R,T):
        x.append(r * np.cos(t))
        y.append(r * np.sin(t))
    return np.array(x), np.array(y)

def createBoxImage(h, w):
    imageHeight = 280
    imageWidth = 180
    y_center = imageHeight / 2
    x_center = imageWidth / 2
    y_start = y_center - (h/2)
    x_start = x_center - (w/2)
    nArray = np.zeros((imageHeight, imageWidth), np.float32)
    for i in range(h):
        for j in range(w):
            nArray[y_start + i][x_start + j] = 255.0
    cv2.imwrite("greyBoxImage.bmp", nArray)
    return nArray

class Snake:
    im = None
    gray = None
    blur = None
    cropped = None
    dx = None
    dy = None
    ddx = None
    ddy = None
    alpha = None
    beta = None
    lambdaa = None
    blur_amount = None
    matrix_A = None
    # template = None
    templatex = None
    templatey = None
    n = None
    pref = None

    def __init__(self,templatex,templatey,alpha,beta,lambdaa,blur_amount,image = None):
        self.alpha = alpha
        self.beta = beta
        self.lambdaa = lambdaa
        self.blur_amount = blur_amount
        self.templatex = templatex
        self.templatey = templatey
        self.n = len(self.templatex)


        self.im = image
        hsv = cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,0,100), (0,255,255))
        mask2 = cv2.inRange(hsv, (117,0,100), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2 )
        cropped = cv2.bitwise_and(image, image, mask=mask)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_HSV2BGR)
        ret,self.cropped = cv2.threshold(cropped,10,255,cv2.THRESH_BINARY)
        self.cropped = np.invert(self.cropped)
        self.gray =  cv2.cvtColor(self.cropped, cv2.COLOR_BGR2GRAY)
        self.im = image
        self.blur = cv2.GaussianBlur(self.gray,(self.blur_amount,self.blur_amount),0)
        if(self.blur_amount>30):
            ret,self.blur = cv2.threshold(self.blur,100,255,cv2.THRESH_BINARY)
            self.blur = cv2.GaussianBlur(self.gray,(25,25),0)
        giy, gix = np.gradient(self.blur)
        gmi = (gix**2 + giy**2)**(0.5)
        gmi = (gmi - gmi.min()) / (gmi.max() - gmi.min())
        self.ddy, self.ddx = np.gradient( gmi )
        self.matrix_A = self.getMatrix_A(self.alpha,self.beta,self.n)
        self.pref = np.linalg.inv(np.eye(self.n)-self.lambdaa*self.matrix_A)


    def getMatrix_B(self):
        n=self.n
        B_x = np.zeros(n)
        B_y = np.zeros(n)
        for i in range(0,n):
            B_x[i] = self.ddx[int(self.templatey[i])-1][int(self.templatex[i])-1]
            B_y[i] = self.ddy[int(self.templatey[i])-1][int(self.templatex[i])-1]
        return B_x,B_y

    
    def getMatrix_A(self,alpha,beta,n):
        matrix=np.zeros((n,n))
        row = np.zeros(n)
        row[0] = (-2*alpha - 6*beta)
        row[1] = (1*alpha + 4*beta)
        row[2] = -beta
        row[n-1] = (1*alpha + 4*beta)
        row[n-2] = -beta
        for i in range(0,n):
            matrix[i] = row
            row = np.roll(row, 1)

        return matrix



    def getDelta(self):
        # deltaX = lambdaa * normalize(matrix_A.dot(np.transpose(tempx)) + Bx)
        Bx,By = self.getMatrix_B(self.templatex,self.templatey)
        # matrix_A = self.getMatrix_A(self.alpha,self.beta,len(tempx))
        Dx = normalize(self.matrix_A.dot(np.transpose(self.templatex)) + Bx)
        Dy = normalize(self.matrix_A.dot(np.transpose(self.templatey)) + By)
        return Dx,Dy

    def newValue(self):
        self.templatex = np.clip(self.templatex,0,self.im_size(1))
        self.templatey = np.clip(self.templatey,0,self.im_size(0))
        Bx,By = self.getMatrix_B()
        x_part = self.templatex+self.lambdaa*Bx
        y_part = self.templatey+self.lambdaa*By
        self.templatex = self.pref.dot(x_part)
        self.templatey = self.pref.dot(y_part)
        # return self.pref.dot(x_part),self.pref.dot(y_part)

    def show_im(self):
        # plt.imshow(self.gray,cmap='gray')
        plt.imshow(self.im)

    def im_size(self,index):
        return np.size(self.gray, index)
    
    def save_im(self,rootbase,idx,extension):
        plt.savefig('output_'+rootbase[:-1]+'/' + rootbase+"%05d"%idx+extension,dpi=300, bbox_inches='tight')

    def save_all(self):
        sp.misc.imsave('im.jpg', self.im)
        sp.misc.imsave('gray.jpg', self.gray)
        sp.misc.imsave('blur.jpg', self.blur)
        sp.misc.imsave('cropped.jpg', self.cropped)
        # sp.misc.imsave('dx.jpg', self.dx)
        # sp.misc.imsave('dy.jpg', self.dy)
        sp.misc.imsave('ddx.jpg', self.ddx)
        sp.misc.imsave('ddy.jpg', self.ddy)

    def getTemplates(self):
        return self.templatex,self.templatey
    def gettempx_30(self):
        return self.templatex[30]   

def transform(tempx,tempy,x,y):
    rangex = np.amax(tempx) - np.amin(tempx)
    rangey = np.amax(tempy) - np.amin(tempy)
    dx = x[1][0]-x[0][0]
    dy = x[1][1]-x[0][1]
    distancex = math.sqrt(math.pow(dx,2)+math.pow(dy,2))
    ydx = y[1][1]-y[0][1]
    ydy = y[1][0]-y[0][0]
    distancey = math.sqrt(math.pow(ydx,2)+math.pow(ydy,2))
    scalex = distancex/rangex
    scaley = distancey/rangey
    tempx = tempx*scalex
    tempy = tempy*scaley
    angle = -1*(math.degrees(math.atan2(-dy, -dx)) - 180)
    array = rotate(tempx,tempy,angle)
    rx=[]
    ry=[]
    for pair in array:
        rx.append(pair[0])
        ry.append(pair[1])
    tempx = rx + (x[0][0]+x[1][0])/2
    tempy = ry + (y[0][1]+y[1][1])/2
    return tempx,tempy

tempx_30 = []

def main(directory,root,idx1,idx2,lip_template):
    templatex = []
    templatey = []
    for pair in lip_template:
        templatex.append(pair[0])
        templatey.append(pair[1])
    tempx = np.array(templatex)
    tempy = np.array(templatey)
    rootbase = root.rsplit( ".", 1 )[ 0 ]
    rootbase = rootbase[:-5]
    type = root.rsplit( ".", 1 )[ 1 ]
    extension = "." + type
    if(directory[-1]!="/"):
        directory = directory + "/"
    filename = directory +rootbase+"%05d"%idx1+extension
    # shutil.rmtree('output_'+rootbase[:-1])
    os.mkdir('output_'+rootbase[:-1])
    im = plt.imread(filename)
    # plt.imshow(im)
    alpha = .000001 #.000001
    beta = .001 #.001
    lambdaa=35
    check = 0
    for idx in range(idx1,idx2+1):
        filename = directory+rootbase+"%05d"%idx+extension
        im = plt.imread(filename)
        snake = Snake(tempx,tempy,alpha,beta,lambdaa,11,im)
        plt.clf()
        for iter in range(1001):
            snake.newValue()
        tempx,tempy = snake.getTemplates()
        snake = Snake(tempx,tempy,alpha,beta,15,5,im)
        for iter in range(100):
            snake.newValue()
        tempx,tempy = snake.getTemplates()
        print idx
        tck, u = interpolate.splprep([tempx, tempy], s=0, per=True)
        tempx, tempy = interpolate.splev(np.linspace(0, 1, 33), tck)
        snake.show_im()
        # plt.scatter(tempx, tempy, c='r', s=5)
        plt.plot(tempx,tempy, '-b')
        # plt.plot(tempx_lefteye,tempy_lefteye, '-b')
        # plt.plot(tempx_righteye,tempy_righteye, '-b')
        
        plt.axis('off')
        snake.save_im(rootbase,idx,extension)
        meanx = np.mean(tempx)
        meany = np.mean(tempy)
        addx=tempx-meanx
        addy=tempy-meany
        addx*=1.0
        addy*=1.0
        tempx=meanx+addx
        tempy=meany+addy

    return 
    
    




template_lip2 = [(632.71554073, 456.1141759), (646.57983638, 446.54508913), (660.44413205, 436.97600237), (672.5168735, 430.90084683), (684.58961495, 424.8256913), (696.80940264, 420.40480598), (709.02919033, 415.98392066), (720.38111145, 418.15670338), (731.73303257, 420.32948611), (741.2375784, 416.30279521), (750.74212424, 412.27610431), (767.72139028, 414.10151819), (784.70065633, 415.92693207), (797.65567522, 419.77739779), (810.61069412, 423.62786352), (826.09249843, 432.25574905), (841.57430274, 440.88363459), (836.34668634, 453.0196985), (831.11906994, 465.15576241), (823.6587606, 475.82294694), (816.19845127, 486.49013146), (801.89301688, 491.0964076), (787.5875825, 495.70268375), (769.4049472, 503.98828194), (751.22231191, 512.27388013), (732.15739917, 510.63385707), (713.09248642, 508.99383401), (697.90477459, 503.67448889), (682.71706277, 498.35514377), (671.84769056, 494.31928724), (660.97831837, 490.2834307), (651.90050037, 482.75364292), (642.82268239, 475.22385514)]
main("testimages/liptracking2/","liptracking2_00068.jpg",1302, 1602, template_lip2)



template_lip4 = [(818.78035718, 616.60794576), (835.1149495, 603.90885915), (851.44954183, 591.20977257), (865.39774301, 583.65615308), (879.34594417, 576.1025336), (893.25301804, 571.13993447), (907.1600919, 566.17733534), (919.3127321, 571.31601068), (931.46537229, 576.45468602), (942.35409751, 571.68174663), (953.24282273, 566.90880726), (971.59366045, 572.38343673), (989.94449817, 577.8580662), (1003.64593549, 585.85056883), (1017.34737282, 593.84307147), (1033.22956477, 609.64520875), (1049.11175671, 625.44734605), (1041.91145594, 643.47476772), (1034.71115517, 661.50218941), (1025.24784515, 676.90201715), (1015.78453515, 692.30184487), (999.57332474, 697.22787041), (983.36211434, 702.15389595), (962.46037624, 712.188815), (941.55863814, 722.22373404), (920.90366388, 716.71253097), (900.24868961, 711.20132791), (884.28424305, 700.58123133), (868.3197965, 689.96113474), (856.9224957,
682.0052057), (845.52519492, 674.04927667), (836.51428529, 660.94788051), (827.50337566, 647.84648437)]
main("testimages/liptracking4/","liptracking4_00068.jpg",68, 338, template_lip4)



template_lip3 = [(485.09754256, 282.86520747), (495.00933641, 276.60135201), (504.92113027, 270.33749654), (513.45440849, 266.51661386), (521.9876867, 262.69573118), (530.55009652, 260.07895451), (539.11250633, 257.46217783), (546.79500011, 259.57549165), (554.47749389, 261.68880548), (561.16278846, 259.22788387), (567.84808302, 256.76696227), (579.4005769, 258.89708805), (590.95307077, 261.02721382), (599.66113856, 264.43096716), (608.36920636, 267.83472051), (618.60144779, 274.81603108), (628.83368921, 281.79734165), (624.72727388, 290.33036601), (620.62085856, 298.86339037), (615.0485328, 306.22706951), (609.47620704, 313.59074864), (599.47701839, 316.24228609), (589.47782975, 318.89382355), (576.66334662, 324.02309457), (563.8488635, 329.15236558), (550.85959079, 327.05700058), (537.87031809, 324.96163558), (527.69633985, 320.38853702), (517.52236163, 315.81543845), (510.25107266, 312.37692433), (502.9797837, 308.93841021), (497.08701038, 303.0569233), (491.19423706, 297.1754364)]
main("testimages/liptracking3/","liptracking3_01295.jpg",1295, 1595, template_lip3)