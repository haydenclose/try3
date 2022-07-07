# -*- coding: yutf-8 -*-
"""
Created on Tue Mar 10 10:07:58 2020

@author: RA05
"""

import os
os.chdir(os.path.dirname(os.path.abspath('Imagery_process_main.py')))

import pandas as pd
import configparser
import time
import glob
import cv2
import Image_white_balance_functions as WB
import Photo_suitability_functions as ps
import Callums_FOV_functions as cf
from ast import literal_eval
from PIL import Image
config = configparser.ConfigParser()
config.read('README_User_input.INI')

#*****************************INPUT PARAMETERS*****************************

src = config['Folders']['src']
txt_src = config['Folders']['txt_src']
dst = config['Folders']['dst']
if not os.path.exists(dst):
    os.makedirs(dst)
    
unsuitable = config['Folders']['unsuitable']
if not os.path.exists(unsuitable):
    os.makedirs(unsuitable)   
    
ColourCorrect = config['Folders']['ColourCorrect']
if not os.path.exists(ColourCorrect):
    os.makedirs(ColourCorrect)     

identifiedlaserimgs = config['Folders']['identifiedlaserimgs']
if not os.path.exists(identifiedlaserimgs):
    os.makedirs(identifiedlaserimgs)   

dst_reassess = config['Folders']['dst_reassess']
if not os.path.exists(dst_reassess):
    os.makedirs(dst_reassess)
    
dst_tables = config['Folders']['dst_tables']
if not os.path.exists(dst_tables):
    os.makedirs(dst_tables)

blur_threshold = float(config['Blur and pixel thresholds']['blur_threshold'])
px_too_dark = int(config['Blur and pixel thresholds']['pixels_too_dark'])
px_too_bright = int(config['Blur and pixel thresholds']['pixels_too_bright'])
perc_unusable_img = int(config['Blur and pixel thresholds']['percentage_unusable_image'])
sk_up = float(config['Skewness and Kurtosis']['skewness_upper'])
sk_low = float(config['Skewness and Kurtosis']['skewness_lower'])
ku_up = float(config['Skewness and Kurtosis']['kurtosis_upper'])



if ps.query_yes_no("Do you want to run White Balance alterations? (y/n)") is True:

     WB.imgWBcorrection(src,ColourCorrect)
     src = ColourCorrect
###*************************BLUR AND ALTIMETRY CHECK*****************************

if ps.query_yes_no("Do you want to run BLUR AND ALTIMETRY CHECKS before the QUALITY ASSESSMENT? (y/n)") is True:

        df_EXIF_DT = ps.altimetry_blur_filter(src, dst_tables)
        df_EXIF_DT['Datetime'] = pd.to_datetime(df_EXIF_DT['Datetime'].str.strip(), format='%Y:%m:%d %H:%M:%S')

        df_altimetry = ps.cam_altitude(txt_src)
        df_altimetry['Datetime'] = pd.to_datetime(df_altimetry['Datetime']).values.astype('datetime64[s]')
        dfalt_blur = pd.merge(df_EXIF_DT[['Photo_path', 'Photo_ID', 'Datetime', 'blur_value']].sort_values('Datetime'), 
                              df_altimetry[['Datetime', 'Depth(m)', 'Lat','Lon', 'Altitude','Temp','Bearing']].sort_values('Datetime'), 
                              how='left', on='Datetime')
        # Recorded to millisecond so averages the values at the same second
        dfalt_blur = dfalt_blur.groupby(['Photo_path', 'Photo_ID', 'Datetime', 'blur_value']).aggregate({'Depth(m)': 'mean',
                                                                                                         'Lat': 'mean',
                                                                                                         'Lon': 'mean',
                                                                                                         'Altitude': 'mean', 
                                                                                                         'Temp': 'mean',
                                                                                                         'Bearing':'mean'})  
        
        dfalt_blur = dfalt_blur.reset_index(level=(['Photo_path', 'Photo_ID', 'Datetime', 'blur_value']), drop=False) #remove index used to group and create means
        dfalt_blur['suitable?'] = 'Y'
        
        ## CURRENTLY THE ALTITUDE THRESHOLD IS CHECKED HERE - if you don't want it to affect the selcetion, remove the check below
        dfalt_blur.loc[(dfalt_blur['blur_value'] <= blur_threshold) | (dfalt_blur['Altitude'] > 2), 'suitable?'] = 'N'
        
        
        dfalt_blur.to_csv(os.path.join(dst_tables, 'Blur_Altimetry_selection.csv'), index=False)
        
        print('Moving the good photos to the destination folder...')
        for i,row in dfalt_blur.iterrows():
                if row['suitable?'] == 'Y':
                        print (row['Photo_ID'])
                        ps.copyFile(row['Photo_path'], os.path.join(dst, row['Photo_ID']), buffer_size=10485760, perserveFileDate=True)
                                 
                elif row['suitable?'] == 'N':
                        print (row['Photo_ID'])
                        ps.copyFile(row['Photo_path'], os.path.join(unsuitable, row['Photo_ID']), buffer_size=10485760, perserveFileDate=True)
        pass


#*****************************************************************************
#*************************RUN IMAGE IMPROVEMENTS******************************
        ## RAWTHERAPY TO BE RUN PRIOR TO THE QUALITY ASSESSMENT ##
#*****************************************************************************

a=0
#*****************************RUN QUALITY ASSESSMENT**************************
if ps.query_yes_no("Do you want to assess the quality of the photos that passed the first selection?") is True:
        
        a=+1
    
        df_results = ps.photo_quality_assess(dst, dst_reassess, dst_tables, perc_unusable_img, px_too_dark, 
                                             px_too_bright, sk_up, sk_low, ku_up)
        
        #groupby station and count number of x - needs to be checked
        df_results['Sample_ID'] =  df_results['Photo_ID'].str[-26:-8]
        df_samples_count = df_results.groupby('Sample_ID').count()

else:
        pass
    
#******************ASSESS MANUALLY THE UNCERTAIN PHOTOS***********************

if a==1 and ps.query_yes_no("Do you want to assess the Uncertain photos? (this procedure might take some time)") is True:

        df_final_results = ps.show_and_choose(df_results, src, dst_reassess, px_too_dark, px_too_bright)

else:
        pass        



#*************************FOV CALCULATIONS************************************
        #***From this point no .ini - needs to be integrated****
            #double-check variables prior to running#
    
print ("FOV calculations in progress...")

#src = "H:/RandD/20200331_Python_imagery_test/20210430_Rawtherapy_TEST/test_processed//"
# src = "C:/Users/hc05/Documents/MPA_IMAGE_PROCESSING/INPUT_IMG/"
# dst = "C:/Users/hc05/Documents/MPA_IMAGE_PROCESSING/RESULTS/"

save_csv = dst + "Manual_rerun.csv"       
images = glob.glob(src+"*.jpg")   

    
colour ="green"
min_thresh_val = 0.6 #green runs on the red and green bands ratio, usually below 0.7 for laser pointers
detections_allowed = 4
#keep at 2 for 2 point detections
blur_window = 3
area_filter = 19
    
                
laser_width = 22 #cm

mode = "auto"

#PROCESSING

#images=images[0:5]
start = time.time()     

 
if mode == "manual":
    
    images = cf.processList(images, save_csv)   
    df=cf.manual_detectLaser(images,colour,min_thresh_val,detections_allowed,
                          blur_window,area_filter,save_csv)     
    df_filter = cf.clean2pointDetections(df, laser_width)
       
elif mode=="auto":       
        
    df=cf.multiproc_autoLaser(images,colour,min_thresh_val,detections_allowed,
                              blur_window,area_filter) 
    df_filter = cf.clean2pointDetections(df, laser_width)
    print (df)
      
else:  
    df_filter = cf.clean2pointDetections(save_csv, laser_width)

   
cf.multi_processImages(src,dst,laser_width,df_filter)    


finish = str((time.time() - start)/60)[0:4]
print("Processed {} images in {} minutes".format(len(images),finish))


########THIS BIT BELOW PLOTS THE LASER POINTS ON THE PHOTOS######
df_filter_MatchingLasers = df_filter[(df_filter["error"] == "NO")] 
df_filter_MatchingLasers['point_x'] = [literal_eval(x) for x in df_filter_MatchingLasers['point_x']]
df_filter_MatchingLasers['point_y'] = [literal_eval(x) for x in df_filter_MatchingLasers['point_y']]
for i, row in df_filter_MatchingLasers.iterrows():
        print (row[1])
        path = os.path.join(src, row[1])
        
        im_col= cv2.imread(path)
        clean_im = im_col.copy()
    
        for x, y in zip(row[3], row[4]):
            
                cv2.circle(clean_im,(x,y ), 8, (0,0,255), -1)
        #First measure point        
        label = (int(row[7]),int(row[9]))
        cv2.putText(clean_im,'First Point',label,cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 8)  
        
        #Second point
        label = (int(row[8]),int(row[10]))
        cv2.putText(clean_im,'Second Point',(label),cv2.FONT_HERSHEY_SIMPLEX,2, (0, 0, 255), 8)  
        
        # cv2.imwrite doesnt save exif data so use pil using method below        
        imWithEXIF = Image.open(path)                         # Read your original image using PIL/Pillow
        imWithEXIF.info['exif']                               # Read Exif data and atttach to image
        corrected = cv2.cvtColor(clean_im, cv2.COLOR_BGR2RGB) # CV2 reads it different to pil so change to match
        OpenCVImageAsPIL = Image.fromarray(corrected)         # Convert OpenCV image to PIL Image
        OpenCVImageAsPIL.save(identifiedlaserimgs + row[1], format='JPEG', exif=imWithEXIF.info['exif']) # Enc
        
   

########FINALLY, THESE TWO LINES MERGE DF_FILTER WITH THE BLUR DATAFRAME###### 
                    ##export to csv is missing##

df_filter = df_filter.rename(columns={"img": "Photo_ID"})
df_ultimate = pd.merge(dfalt_blur, df_filter[['Photo_ID','error', 'pixel_x_cm', 'FOV_m2']], how='left', on='Photo_ID')
