# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:43:24 2020

@author: RA05
"""

import cv2
import numpy as np
from scipy.spatial import distance as dist
import math
import pandas as pd
import itertools
from math import acos, degrees
#from shapely import geometry
import matplotlib.pyplot as mp
#from shapely.geometry.polygon import LinearRing, Polygon
import os
from multiprocessing.dummy import Pool as ThreadPool 
import multiprocessing
from random import randint
from math import sqrt
import imutils

def squared_distance(p1, p2):
    # TODO optimization: use numpy.ndarrays, simply return (p1-p2)**2
    sd = 0
    for x, y in zip(p1, p2):
        sd += (x-y)**2
    return sd


def get_proximity_matrix(points, threshold):
    n = len(points)
    t2 = threshold**2
    # TODO optimization: use sparse boolean matrix
    prox = [[False]*n for k in range(n)]
    for i in range(0, n):
        for j in range(i+1, n):
            prox[i][j] = (squared_distance(points[i], points[j]) < t2)
            prox[j][i] = prox[i][j]  # symmetric matrix
    return prox


def find_clusters(points, threshold):
    n = len(points)
    prox = get_proximity_matrix(points, threshold)
    point_in_list = [None]*n
    clusters = []
    for i in range(0, n):
        for j in range(i+1, n):
            if prox[i][j]:
                list1 = point_in_list[i]
                list2 = point_in_list[j]
                if list1 is not None:
                    if list2 is None:
                        list1.append(j)
                        point_in_list[j] = list1
                    elif list2 is not list1:
                        # merge the two lists if not identical
                        list1 += list2
                        point_in_list[j] = list1
                        del clusters[clusters.index(list2)]
                    else:
                        pass  # both points are already in the same cluster
                elif list2 is not None:
                    list2.append(i)
                    point_in_list[i] = list2
                else:
                    list_new = [i, j]
                    for index in [i, j]:
                        point_in_list[index] = list_new
                    clusters.append(list_new)
        if point_in_list[i] is None:
            list_new = [i]  # point is isolated so far
            point_in_list[i] = list_new
            clusters.append(list_new)
    return clusters


def average_clusters(points, threshold=1.0, clusters=None):
    if clusters is None:
        clusters = find_clusters(points, threshold)
    newpoints = []
    for cluster in clusters:
        n = len(cluster)
        point = [0]*len(points[0])  # TODO numpy
        for index in cluster:
            part = points[index]  # in numpy: just point += part / n
            for j in range(0, len(part)):
                point[j] += int(part[j] / n)  # TODO optimization: point/n later
        newpoints.append(point)
    return newpoints


def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


def contourCentroids(contours):
    
    
    '''Gets centroids of contours'''


    centroid_x=[]
    centroid_y=[]
    areas=[]
    for cnt in contours:
    
        try:   
            m=cv2.moments(cnt)      
            cx=int(m['m10']/m['m00'])
            cy=int(m['m01']/m['m00']) 
            centroid_x.append(cx)
            centroid_y.append(cy) 
            
            area = cv2.contourArea(cnt)
            areas.append(area)
        except ZeroDivisionError:
            pass
        
        
    return centroid_x, centroid_y ,areas



def laserDetection(colour_image,laser_colour,min_thresh_value,
                   detections_allowed, blur_window,
                   area_filter):
    
    ''' Function to detect lasers within an image. Configurable to 
        red/green lasers, custom threshold values can be applied ,
        number of detections and more'''

    if  min_thresh_value ==None and laser_colour =="Red":
        min_thresh_value = 230
        
    elif min_thresh_value ==None and laser_colour =="green":
        min_thresh_value = 0.6
        
        
    if detections_allowed == None:
        detections_allowed = 10
    
    if blur_window ==None:
        blur_window = 3
        
    if area_filter ==None:
        area_filter = 25        

    #colour_image= cv2.imread(img)        
        
    b,g,r = cv2.split(colour_image)
    np.seterr(divide='ignore', invalid='ignore')
    rg = r / g
    #split image into 3 bands . If red lasers we will focus on red band else 
    #change to green - RA addition: for green lasers the ratio between red and green bands is used
    ## not found laser_colour
    if laser_colour =="Red":
        
        blurred = cv2.GaussianBlur(r, (blur_window, blur_window), 0)
        
    else:
        blurred = cv2.GaussianBlur(rg, (blur_window, blur_window), 0)
        
      
    #basic blur operation
    #thresh = cv2.threshold(blurred, min_thresh_value, 255,
                           #cv2.THRESH_BINARY)[1]
    
    thresh = cv2.threshold(blurred, min_thresh_value, 255, 
                           cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=1)
    thresh = thresh.astype(np.uint8)
    #change first value to alter sensitivity (low value = high detections ,
    #high value = low detections)
    
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_SIMPLE)
    
    centroid_x, centroid_y, area = contourCentroids(contours)   
    
    data = pd.DataFrame({"x":centroid_x,"y":centroid_y,"area":area})  
    #detection coords added to dataframe and band value is 
    #added to coords and sorted(high to low)
    
    if laser_colour =="Red":    
        data["val"] = r[data["y"],data["x"]]
        data["green"] = g[data["y"],data["x"]]
        data["blue"] = b[data["y"],data["x"]]
        
    else:
        data["val"] = g[data["y"],data["x"]]
        data["red"] = g[data["y"],data["x"]]
        data["blue"] = b[data["y"],data["x"]]        
#df[(df.depth <3500)        
    
    data = data[(data.area >area_filter)]
    
    #data = data[(data.red/data.val < 0.5)]
    #data = data[(data.green < 220) & (data.blue < 220)]

    
    data = data.sort_values("val",ascending=False)
    
    centroid_x = data["x"].tolist()[0:detections_allowed]
    centroid_y = data["y"].tolist()[0:detections_allowed]
 
    points = list(zip(centroid_x, centroid_y))

    threshold = 200.0
    clusters = find_clusters(points, threshold)
    clustered = average_clusters(points, clusters=clusters)
    
    centroid_x, centroid_y = list(map(list, zip(*clustered)))
    
    

    #print ("clusters:", clusters)
    #print (clustered)

    #import matplotlib.pyplot as plt
    #ax = plt.figure().add_subplot(1, 1, 1)
    #for cluster in clustered:
        #ax.add_patch(plt.Circle(cluster, radius=threshold/2, color='g'))
    #for point in points:
        #ax.add_patch(plt.Circle(point, radius=threshold/2, edgecolor='k', facecolor='none'))
    #plt.plot(*zip(*points), marker='o', color='r', ls='')
    #plt.plot(*zip(*clustered), marker='.', color='g', ls='')
    #plt.axis('equal')
    #plt.show()

    #for x, y in zip(centroid_x,centroid_y):
        #cv2.circle(colour_image,(x,y ), 8, (0,0,255), -1)
    #plots circles of detections on image    
        #cv2.drawContours(colour_image, contours, -1, (0),3)
    
#    showImage(colour_image)
#    showImage(blurred)
#    showImage(thresh)
    
    return centroid_x, centroid_y




def processList(input_images, save_csv):
    
    '''Creates a detections csv for manual laser operations.Also checks to see
       if a csv already exsisits so can resume from previous sessions'''

    if not os.path.exists(save_csv):
        df=pd.DataFrame({"img":[],"error":[],"point_x":[],"point_y":[], "image_h":[], "image_w":[]})
        df.to_csv(save_csv,index=False)

    processed_imgs = pd.read_csv(save_csv)["img"].tolist()
    
    imgs=[x for x in input_images if os.path.basename(x) not in processed_imgs]
    
    
    return imgs



def auto_detectLaser(img,laser_colour,min_thresh_value, detections_allowed,
                     blur_window,area_filter):
    
    '''Runs automatic laser detection on an image, returns laser positions'''
    
    try:
        im_col= cv2.imread( img,1)
        
        mask = np.zeros(im_col.shape,np.uint8)

        mask[500:4500,1000:4000] = im_col[500:4500,1000:4000]
        #img = images[2]
        #mask = im_col
        #colour_image= mask
        # min_thresh_value =0.6
        #change mask indicies if uisng different size images or after 4 lasers
        
        detections_x, detections_y = laserDetection(mask,laser_colour, 
                                                    min_thresh_value,
                                                    detections_allowed,
                                                    blur_window,
                                                    area_filter)                     
        error="NO"    
        
    except Exception as e:            
        error = "YES"  
        detections_x = 'NaN'
        detections_y = 'NaN'
    
    height, width, channels = im_col.shape
    info = pd.DataFrame({"img":os.path.basename(img),
                         "error":error,
                         "point_x":[detections_x],
                         "point_y":[detections_y],
                         "image_h":[height],
                         "image_w":[width]})
    
    return info




def multiproc_autoLaser(images,colour,min_thresh_val,detections_allowed,
                   blur_window,area_filter):
    
    '''Runs auto laser detection across multiple cores'''


    input_list = [(img,colour,min_thresh_val,detections_allowed,
                       blur_window,area_filter) for img in images]
    
    
    pool = ThreadPool(multiprocessing.cpu_count())
    data = pool.starmap(auto_detectLaser,input_list)
    pool.close()
    pool.join()
    
    detections = pd.concat(data)
    
    return detections


def manual_detectLaser(images,colour,min_thresh_val,detections_allowed,
                   blur_window,area_filter,save_csv):
    
    '''Runs manual laser detection on an image, saves laser positions to csv'''
    img = images[1]
    for img in images:
        
        try:
        
            im_col= cv2.imread( img,1)
            
            height, width, channels = im_col.shape
            
            mask = np.zeros(im_col.shape,np.uint8)

            mask[500:4500,1000:4000] = im_col[500:4500,1000:4000]
#change mask indicies if uisng different size images or after 4 lasers
            im_col=mask.copy()
            
            clean_im = im_col.copy()
            clean_im2=im_col.copy()
            
            detections_x, detections_y, area = laserDetection(im_col,colour, 
                                               min_thresh_val,
                                               detections_allowed,
                                               blur_window,
                                               area_filter) 
            
            for x, y in zip(detections_x,detections_y):
                cv2.circle(clean_im,(x,y ), 8, (0,0,255), -1) 
            
            
            showImage(clean_im)
            man_laz = manual_detection(clean_im2)
            
            man_x=[x[0] for x in man_laz]
            man_y=[y[1] for y in man_laz] 
            error="NO"
            
            if not man_x or  not man_y :
                pass
            
            else:
                detections_x = man_x
                detections_y = man_y
            
        except Exception as e:
            error="YES"
            
        info = pd.DataFrame({"img":os.path.basename(img),
                         "error":error ,
                         "point_x":[detections_x],
                         "point_y":[detections_y],
                         "image_h":[height],
                         "image_w":[width]}) 
           
            
        info.to_csv(save_csv,index=False,mode='a', header=False)
        
        return info

def distance(x1, x2):
    if x1 >= x2:
        result = x1-x2
    else:
        result = x2-x1
    return result
def clean2pointDetections(data, laser_width): 
    
    '''This cleans a df or csv of detections for 2 point laser detections.
       will require modification for 4 lasers'''

    if isinstance(data,str):
                          
        df = pd.read_csv(data)
        
    else:
        df = data
    
    df["point_x"]= df["point_x"].astype(str)
    df["point_y"] = df["point_y"].astype(str)
    
    ######HERE 12 needs changing to 18
    df_filter =df[(df["point_x"].str.len()>=12)|(df["point_y"].str.len()>=12)]
    df_lessthan2lasers =df[(df["point_x"].str.len()<12)|(df["point_y"].str.len()<12)]
    df_lessthan2lasers["error"] = "YES"
    #removes any rows which dont have valid coordinates
    #if adapting for 4 point routines will need changed to keep 4 coord
    
    df_filter = df_filter.replace('[\[\]]',"",regex=True)
    #removes nonsense brackets
    
    
    #Section below loops through the lasers so to get two which are parrallel(with a tollerance of 300 pixels) 
   #to create the FoV so not lasers diaganal are not used
    row = 3
    YY = 0
    df_filter["x0"], df_filter["x1"], df_filter["y0"], df_filter["y1"]= '','','',''
    for row in range(len(df_filter)):
        XPOINTS = df_filter["point_x"].iloc[row]
        XPOINTS = XPOINTS.split(", ")
        XPOINTS = [int(n) for n in XPOINTS]
        YPOINTS = df_filter["point_y"].iloc[row]
        YPOINTS = YPOINTS.split(", ")
        YPOINTS = [int(n) for n in YPOINTS]
        level_lasers ='No'
        for YY in range(len(YPOINTS)):
            if YY != len(YPOINTS)-1:
               if distance(YPOINTS[YY],YPOINTS[YY+1])<500 and level_lasers =='No' and len(YPOINTS)>=2:
                  df_filter["x0"].iloc[row] = XPOINTS[YY]
                  df_filter["x1"].iloc[row] = XPOINTS[YY+1]
                  df_filter["y0"].iloc[row] = YPOINTS[YY]
                  df_filter["y1"].iloc[row] = YPOINTS[YY+1]
                  level_lasers =='Yes'
               elif  level_lasers =='No' and YY+2<=len(YPOINTS)-1:
                      if distance(YPOINTS[YY],YPOINTS[YY+2])<500:
                         df_filter["x0"].iloc[row] = XPOINTS[YY]
                         df_filter["x1"].iloc[row] = XPOINTS[YY+2]
                         df_filter["y0"].iloc[row] = YPOINTS[YY]
                         df_filter["y1"].iloc[row] = YPOINTS[YY+2]
                         level_lasers =='Yes'
               elif  level_lasers =='No' and YY+3<=len(YPOINTS)-1:
                   if distance(YPOINTS[YY],YPOINTS[YY+3])<500:
                       df_filter["x0"].iloc[row] = XPOINTS[YY]
                       df_filter["x1"].iloc[row] = XPOINTS[YY+3]
                       df_filter["y0"].iloc[row] = YPOINTS[YY]
                       df_filter["y1"].iloc[row] = YPOINTS[YY+3]
                       level_lasers =='Yes'
               else:
                    pass
    cols = ["x0","x1","y0","y1"]
    df_filter = df_filter.replace('',0)
    df_filter[cols] = df_filter[cols].astype(str).astype(int)
   
    df_filter_noMatchingLasers = df_filter[(df_filter["x0"] == 0)]  
    df_filter_noMatchingLasers["error"] = "YES"
    df_filter=df_filter[df_filter!=0].dropna()
        
    # df_filter["dodgy"] = np.where(df_filter["x0"] <= 4000,0,1)
    df_filter.reset_index(inplace=True)
    
    df_filter["pixel_x_cm"] = df_filter.apply(lambda x: Euclidistance(x.x0, x.y0, x.x1, x.y1, laser_width), axis=1)
    df_filter["FOV_m2"] = df_filter.apply(lambda x: ((x.image_h/x.pixel_x_cm)*(x.image_w/x.pixel_x_cm)/10000), axis=1)
    
    df_all = pd.concat([df_filter,df_filter_noMatchingLasers,df_lessthan2lasers], ignore_index=True, sort=False)
    
    return df_all



def processImages(input_dir,output_folder,laser_cm,df,my_idx):
    
    '''This function processes laser detections which are within a csv or df'''
      
    
    image_name=[]
    dodge=[]
    error=[]
    centx=[]
    centy=[]
    pixels=[]
    width=[]  
    
    try:
                       
        im_path = input_dir + df["img"]    
        detections_x=[df["x0"].tolist(),df["x1"].tolist()]
        detections_y=[df["y0"].tolist(),df["y1"].tolist()] 
        dodgy =df["dodgy"]
                    
        im_col= cv2.imread( im_path,1)
        clean_im = im_col.copy()
        
    
        for x, y in zip(detections_x, detections_y):
            cv2.circle(clean_im,(x,y ), 8, (0,0,255), -1) 
            
        #plots detcions as circles change tuple vals for alt. colours        
        #uc.showImage(clean_im)
             
        pixels_per_cm ,cx,cy = Euclidistance(detections_x,detections_y,
                                              laser_cm) 
        
        
        #Makes nessc. folders for various outputs
        folders =["na\\","out\\"]

        [os.makedirs(output_folder + i) for i in folders
             if not os.path.exists(output_folder +i)]         
                       
        if dodgy == 1:
                        
            output_folder = output_folder + "na\\"
            
        else:
            
           output_folder = output_folder + "out_\\"
            
       
        #sets relevant ouput folder depending on grid size,dodginess, na
                   
        cv2.imwrite(output_folder + df["img"],clean_im)
              
        image_name.append(df["img"])
        
        dodge.append(dodgy)
        error.append("NO")
        centx.append(cx)
        centy.append(cy)
        pixels.append(pixels_per_cm)
        #width.append(out_image.shape[1])
            
    except Exception as e:
        #any errors in processing append values below
        
        
        image_name.append(df["img"])
        dodge.append("No")
        error.append("YES")
        centx.append(-9999)
        centy.append(-9999)
        pixels.append(-9999)
        #width.append(-9999)

    return image_name, dodge,error,centx,centy,pixels,width





def multi_processImages(image_dir,output_folder,laser_width_cm, df_filter):
    
    '''This applies a multicore version of the ProcessImages function'''
    


    numbers = list(range(0,len(df_filter)))
    
    input_list=[(image_dir,output_folder,
                 laser_width_cm, df_filter.loc[i], i)for i in numbers] 
         
    #this creates a tuple of 6 for every image in the dataframe provided
    #the tuple contains all req input parameters
        
    #This is done to allow processing of the images on multiple cores
    #you can use a standard for loop but will take a long time
        

    pool = ThreadPool(multiprocessing.cpu_count())
    pool.starmap(processImages,input_list) 
    pool.close() 
    pool.join()




def getPolygonCombinations(centroid_x,centroid_y):
    
    '''Gets all potential combinations of polygons from centroid positions'''

    points=list(zip(centroid_x,centroid_y))
    
    polygons=[]
    for i in range(0, len(points)+1):
        
        for subset in itertools.combinations(points, i):
            #print (len(subset))
            if len(subset) == 4:
                polygons.append([subset[0],subset[1],subset[2],subset[3]])
            else:
                pass
    #if len(point_collection) <4 skip    
    return polygons



def Euclidistance(centroid_x0, centroid_y0, centroid_x1, centroid_y1, laser_width):
    
    
    '''Draws grid on image based on centroid position and additional params
    (i.e laser locations,real world laser distance,grid size, intervals etc.'''

        
    dist_pixels = dist.euclidean((centroid_x0, centroid_y0), 
                                 (centroid_x1, centroid_y1))

    #distance in pixels between laser points
       
    pixels_per_cm = dist_pixels/laser_width
    
    #x = int((centroid_x[0] + centroid_x[1])/2)
    #y = int((centroid_y[0] + centroid_y[1])/2)
    
    #print(bot_left,top_right)
    
    return  pixels_per_cm
    



def randomPoints(num,w_min,w_max,h_min,h_max):
    
    '''This fucntion generates a specified number of random points. Min and 
    max filters are applied for width and height'''


    x = [randint(w_min, w_max) for i in range(0, num)]
    y = [randint(h_min, h_max) for i in range(0, num)]
    
    return x, y
    


#def distance(p0, p1):
    
  #  '''Distance from point to point'''
    
 #   return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)




def showImage(image):
    
    '''Function to resize and display images'''

    cv2.namedWindow('images',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('images', 1280,960)
    #cv2.imshow("images", np.hstack([im, thresh]))
    cv2.imshow("images",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def NorthAzimuth(lastx,firstx,lasty,firsty):  
  degBearing = math.degrees(math.atan2((lastx - firstx),(lasty -firsty)))  
  if (degBearing < 0):  
      degBearing += 360.0  
  return degBearing 




def manual_detection(im_col):
    
    '''This function allows the manual capturing of user clicked 
        points upon an image'''


    class CoordinateStore:
        def __init__(self):
            self.points = []
    
        def select_point(self,event,x,y,flags,param):
                if event == cv2.EVENT_LBUTTONDBLCLK:
                    cv2.circle(im_col,(x,y),12,(255,0,0),-1)
                    self.points.append((x,y))
    
    
    #instantiate class
    coordinateStore1 = CoordinateStore()
    
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 1280,960)    
    cv2.setMouseCallback('image',coordinateStore1.select_point)
    
    
    
    while cv2.getWindowProperty('image', 0) >= 0:    
        cv2.imshow('image',im_col)
        k = cv2.waitKey(20) & 0xFF & cv2.waitKey(50)
        if k == 27:
            break
    cv2.destroyAllWindows()
    
    return coordinateStore1.points


def image_rotation(images):
    
    for img in images:
        
        imag= cv2.imread( img,1)
        h, w, cha = imag.shape
        
        if h > w:
            imag = cv2.rotate(imag, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(img, imag)
        else:
            pass  