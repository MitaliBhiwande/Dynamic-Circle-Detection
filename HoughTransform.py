import cv2
import numpy as np


#originalImage = cv2.imread('Input.jpg',1)
#originalImage = cv2.imread('/Users/mitalibhiwande/Downloads/coins1.jpg',0)
originalImage = cv2.imread('/Users/mitalibhiwande/Downloads/coins.jpg',1)
cv2.imshow('Original Image',originalImage)

output = originalImage.copy()

#Applying Gausssian Blur on input image
blurredImage = cv2.GaussianBlur(originalImage,(3,3),0)
cv2.imshow('Gaussian Blurred Image',blurredImage)

#Detecting edges in Image using Canny edge Detector
edgeDetectedImage = cv2.Canny(blurredImage,60,100)
cv2.imshow('Edge Detected Image', edgeDetectedImage)

im = cv2.imread("/Users/mitalibhiwande/Downloads/coins.jpg",cv2.IMREAD_GRAYSCALE)
#im = cv2.imread("Input.jpg",cv2.IMREAD_GRAYSCALE)
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 50
params.maxThreshold = 200


# Filter by Area.
#params.filterByArea = True
#params.minArea = 1500

## Filter by Circularity
#params.filterByCircularity = True
#params.minCircularity = 0.8
#
## Filter by Convexity
#params.filterByConvexity = True
#params.minConvexity = 0.87
#    
## Filter by Inertia
#params.filterByInertia = True
#params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector()

keypoints=detector.detect(im)
#print keypoints.
keys=cv2.drawKeypoints(edgeDetectedImage,keypoints,np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("a", keys)
sizes=[]
for keyPoint in keypoints:
    
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size
    sizes.append(s)
    
print sizes

height,width = edgeDetectedImage.shape
print height,width
radiivalue = int(min(height,width)/2)

accumulator_array = np.zeros(((height,width,radiivalue)))

def fill_accumulator_array(x0,y0,radius):
    x = radius
    y=0
    decision=1-x
    
    while(y<x):
        if(x + x0<height and y + y0<width):
            accumulator_array[ x + x0,y + y0,radius]+=1; # Octant 1
        if(y + x0<height and x + y0<width):
            accumulator_array[ y + x0,x + y0,radius]+=1; # Octant 2
        if(-x + x0<height and y + y0<width):
            accumulator_array[-x + x0,y + y0,radius]+=1; # Octant 4
        if(-y + x0<height and x + y0<width):
            accumulator_array[-y + x0,x + y0,radius]+=1; # Octant 3
        if(-x + x0<height and -y + y0<width):
            accumulator_array[-x + x0,-y + y0,radius]+=1; # Octant 5
        if(-y + x0<height and -x + y0<width):
            accumulator_array[-y + x0,-x + y0,radius]+=1; # Octant 6
        if(x + x0<height and -y + y0<width):
            accumulator_array[ x + x0,-y + y0,radius]+=1; # Octant 8
        if(y + x0<height and -x + y0<width):
            accumulator_array[ y + x0,-x + y0,radius]+=1; # Octant 7
        y+=1
        if(decision<=0):
            decision += 2 * y + 1
        else:
            x=x-1;
            decision += 2 * (y - x) + 1
    
    
edges = np.where(edgeDetectedImage==255)
for i in xrange(0,len(edges[0])):
    x=edges[0][i]
    y=edges[1][i]
    for radius in xrange(int(min(sizes)/2),int(max(sizes)*2)):
        fill_accumulator_array(x,y,radius)
        
print accumulator_array

i=0
j=0
filter3D = np.zeros((30,30,radiivalue))
filter3D[:,:,:]=1

while(i<height-30):
    while(j<width-30):
        filter3D=accumulator_array[i:i+30,j:j+30,:]*filter3D
        max_pt = np.where(filter3D==filter3D.max())
        a = max_pt[0]       
        b = max_pt[1]
        c = max_pt[2]
        b=b+j
        a=a+i
        if(filter3D.max()>90):
            cv2.circle(output,(b,a),c,(255,0,0),3)
        j=j+30
        filter3D[:,:,:]=1
    j=0
    i=i+30
                

cv2.imshow('Detected circle',output)


cv2.waitKey(0)
cv2.destroyAllWindows()
            