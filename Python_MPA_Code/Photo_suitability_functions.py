# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:47:52 2020

@author: RA05
"""

##### IMPORT PACKAGES #####
from __future__ import print_function
from __future__ import division

import sys
import os, re, fnmatch
import glob
import time
from datetime import date

import csv
import Callums_FOV_functions as cf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms

from scipy.stats import skew, kurtosis

import cv2 as cv
from PIL import Image
import shutil
from skimage import measure

from fpdf import FPDF


########## YES OR NO #####################
#http://code.activestate.com/recipes/577058/
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

#### ARITHMETIC FUNCTIONS ####

def roundToHalf(array):
    
        return np.around(array * 2.0) / 2.0

def dms2dd(s):
    
    # example: s = """012°51.34756' S"""
    
    degrees, minutes, direction = re.split('[°\']+', s)
    dd = float(degrees) + float(minutes)/60
    if direction in ('S','W'):
        dd*= -1
        
    return '%.*f' % (7, dd)


########## FASTER FILE COPY ###################
###### https://blogs.blumetech.com/blumetechs-tech-blog/2011/05/faster-python-file-copy.html

def copyFile(src, dst, buffer_size=10485760, perserveFileDate=True):
    '''
    Copies a file to a new location. Much faster performance than Apache Commons due to use of larger buffer
    @param src:    Source File
    @param dst:    Destination File (not file path)
    @param buffer_size:    Buffer size to use during copy
    @param perserveFileDate:    Preserve the original file date
    '''
    #    Check to make sure destination directory exists. If it doesn't create the directory
    dstParent, dstFileName = os.path.split(dst)
    if(not(os.path.exists(dstParent))):
        os.makedirs(dstParent)
    
    #    Optimize the buffer for small files
    buffer_size = min(buffer_size,os.path.getsize(src))
    if(buffer_size == 0):
        buffer_size = 1024
    
    if shutil._samefile(src, dst):
        raise shutil.Error("`%s` and `%s` are the same file" % (src, dst))
    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if shutil.stat.S_ISFIFO(st.st_mode):
                raise shutil.SpecialFileError("`%s` is a named pipe" % fn)
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            shutil.copyfileobj(fsrc, fdst, buffer_size)
    
    if(perserveFileDate):
        shutil.copystat(src, dst)

### COPY FROM ONE FOLDER TO ANOTHER IF FILE IS IN A LIST###

def copy_if_inlist (queried_list, src, dst):

        queried_list = queried_list.tolist()

        Photos_found = []
        for srcs, dirs, files in os.walk(src):
                for fil in files:      
                        if fil.endswith(".JPG") and fil[0:-4] in queried_list:
                                copyFile(os.path.join(srcs, fil), os.path.join(dst, fil))
                                Photos_found.append(fil[0:-4])

        for p in queried_list:
                if p not in Photos_found:
                        print (p)



##### FUNCTIONS FOR IMAGE QUALITY ASSESSMENT #####

def variance_of_laplacian(photo):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
    gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
    blur = cv.Laplacian(gray, cv.CV_64F).var()
    
    return blur


def histograms(photo, px_too_dark, px_too_bright):
    
        histSize = 256
        histRange = (0, 256) # the upper boundary is exclusive
        #hist_w = 512
        hist_h = 400
        #bin_w = int(round( hist_w/histSize ))
        bgr_planes = cv.split(photo)
        accumulate = False

        b_hist = cv.calcHist(bgr_planes, [0], None, [histSize], histRange, accumulate=accumulate)
        g_hist = cv.calcHist(bgr_planes, [1], None, [histSize], histRange, accumulate=accumulate)
        r_hist = cv.calcHist(bgr_planes, [2], None, [histSize], histRange, accumulate=accumulate)

        cv.normalize(b_hist, b_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(g_hist, g_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)
        cv.normalize(r_hist, r_hist, alpha=0, beta=hist_h, norm_type=cv.NORM_MINMAX)

        b_skw, g_skew, r_skew = skew(b_hist), skew(g_hist), skew(r_hist)
        a_skw = (b_skw + g_skew + r_skew)/3
        b_kurt, g_kurt, r_kurt = kurtosis(b_hist), kurtosis(g_hist), kurtosis(r_hist)
        a_kurt = (b_kurt + g_kurt + r_kurt)/3
                  
        b_dark_pixels = max((np.cumsum(b_hist[0:px_too_dark+1]))*100)/max(np.cumsum(b_hist[0:256]))
        #g_dark_pixels = max((np.cumsum(g_hist[0:51]))*100)/max(np.cumsum(g_hist[0:256]))
        #r_dark_pixels = max((np.cumsum(r_hist[0:51]))*100)/max(np.cumsum(r_hist[0:256]))
        
        b_bright_pixels = max((np.cumsum(b_hist[px_too_bright+1:256]))*100)/max(np.cumsum(b_hist[0:256]))
        #g_bright_pixels = max((np.cumsum(g_hist[201:256]))*100)/max(np.cumsum(g_hist[0:256]))
        #r_bright_pixels = max((np.cumsum(r_hist[201:256]))*100)/max(np.cumsum(r_hist[0:256]))

        plt.plot(b_hist, color = 'blue')
        plt.plot(g_hist, color = 'green')
        plt.plot(r_hist, color ='red')
        
        return(b_hist, g_hist, r_hist, a_skw, a_kurt, b_dark_pixels, b_bright_pixels)


def masking(photo,px_too_dark,px_too_bright):
    
        gray = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (11, 11), 0)
        thresh_bright = cv.threshold(blurred, px_too_bright, 255, cv.THRESH_BINARY)[1]
        thresh_dark = cv.threshold(blurred, px_too_dark, 255, cv.THRESH_BINARY_INV)[1]
        thresh = cv.add(thresh_bright, thresh_dark)

        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        thresh = cv.erode(thresh, None, iterations=2)
        thresh = cv.dilate(thresh, None, iterations=2)

        #numer of white (i.e. unusable image) vs black pixels
        rows,cols = thresh.shape
        whitepix_perc = int((cv.countNonZero(thresh)*100) / (rows*cols))   

        labels = measure.label(thresh, connectivity=2, background=0)#neighbors = 8
        mask = np.zeros(thresh.shape, dtype="uint8")
        blobcount = 0
        BlobSize_list = []

        # loop over the unique components
        for label in np.unique(labels):
        #if this is the background label, ignore it
            if label == 0:
                continue
            
    	# otherwise, construct the label mask and count the
        #number of pixels 
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv.countNonZero(labelMask)
    
            #if the number of pixels in the component is sufficiently
            #large, then add it to our mask of "large blobs"
            if numPixels > 1000:
                blobcount += 1
                mask = cv.add(mask, labelMask)
                BlobSize_list.append(numPixels*100/(rows*cols))
        
        try:
            Max_blob = max(BlobSize_list)
        except ValueError:
            Max_blob = 0
            
        return(whitepix_perc, blobcount, Max_blob, thresh)


def parameter_tester(test_src, list_low, list_up):
        
        start = time.time()    
        pdf = FPDF()
        images = glob.glob(test_src+"*.jpg")
        images=images[0:]
        d = {}
        os.makedirs(os.path.join(test_src, 'processing'))
        
        blur_list =[]
        IQA_list = []
        photo_list = []
        df_test = pd.DataFrame()
        
        for i in images:
                pdf.add_page()
                c = 0
                
                print ('Now processing ' + i)
                img = cv.imread(i)
                
                b_hist, g_hist, r_hist, _, _, _, _ = histograms(img, 30, 220)
                IQA_score = IQA_brisque(i)
                blur = variance_of_laplacian(img)
                print ('Blur: ' + str(blur))
                print ('IQA: ' + str(IQA_score))
                
                img = cv.resize(img, (0, 0), None, .20, .20)
                fig,ax = plt.subplots()
                ax.plot(b_hist, color = 'blue')
                ax.plot(g_hist, color = 'green')
                ax.plot(r_hist, color ='red')
                ax.set_ylabel('pixel number')
                ax.set_xlabel('pixel value')
                fig.savefig(os.path.join(test_src, 'processing', os.path.basename(i)[0:-4]+'_plot.jpg'))
                
                for l, u in zip(list_low, list_up):
                        _, _, _, thresh = masking(img, l, u)
                        thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
                        font = cv.FONT_HERSHEY_SIMPLEX 
                        cv.putText(thresh, '{} : {}'.format(l, u), (10,30), font, 1, (0, 0, 255), 2, cv.LINE_AA)
                        d["threshold{0}".format(c)] = np.hstack((img, thresh))
                        c+=1
                
                numpy_vertical = np.vstack((d['threshold0'], d['threshold1'], d['threshold2']))
                
                cv.imwrite(os.path.join(test_src, 'processing', os.path.basename(i)[0:-4]+'_combo.jpg'), numpy_vertical)
                
                blur_list.append(round(blur, 2))
                IQA_list.append(round(IQA_score, 2))
                photo_list.append(os.path.basename(i)[0:-4])
                
                pdf.set_font('Arial', 'B', 16)
                pdf.cell(10, 10, os.path.basename(i)[0:-4])
                pdf.image(os.path.join(test_src, 'processing', os.path.basename(i)[0:-4]+'_combo.jpg'),10,30,150,150)
                pdf.set_font('Arial', '', 11)
                pdf.set_y(180)
                pdf.cell(5, 5, 'Blur value: {}'.format(round(blur, 2)))
                pdf.set_y(186)
                pdf.cell(5, 5, 'No-reference image quality score: {}'.format(round(IQA_score, 2)))
                pdf.image(os.path.join(test_src, 'processing', os.path.basename(i)[0:-4]+'_plot.jpg'),10,200,100)

                d.clear()
                fig.clf()

        finish = str((time.time() - start)/60)[0:4]
        print('Processed {} images in {} minutes'.format(len(images),finish))   
        df_test['Photo_ID'] = photo_list
        df_test['Blur_value'] = blur_list
        df_test['IQA_score'] = IQA_list
        
        pdf.add_page(orientation = 'L')
        fig2, ax2 = plt.subplots()
        fig2.patch.set_visible(False)
        ax2.axis('off')
        ax2.axis('tight')
        tab = ax2.table(cellText=df_test.values, colLabels=df_test.columns, loc='left')
        tab.auto_set_column_width(col=list(range(len(df_test.columns))))
        tab.auto_set_font_size(False)
        # draw canvas
        plt.gcf().canvas.draw()
        # get bounding box of table
        points = tab.get_window_extent(plt.gcf()._cachedRenderer).get_points()
        # add 10 pixel spacing
        points[0,:] -= 10; points[1,:] += 10
        # get new bounding box in inches
        nbbox = matplotlib.transforms.Bbox.from_extents(points/plt.gcf().dpi)
        plt.savefig(os.path.join(test_src, 'processing', 'result_table.jpg'), bbox_inches=nbbox, )
        pdf.image(os.path.join(test_src, 'processing', 'result_table.jpg'),10,10)
        
        pdf.output(os.path.join(test_src,'{}_tester_results.pdf'.format(str(date.today()))), 'F')
        df_test.to_csv(os.path.join(test_src,'{}_tester_table_results.csv'.format(str(date.today()))))
        shutil. rmtree(os.path.join(test_src, 'processing'))

   

def show_and_choose(df_results, src, dst,px_too_dark,px_too_bright):
    
        for i, row in df_results.loc[(df_results['suitable?'] == 'Uncertain')].iterrows():
            path = os.path.join(src, row[0])
            photo_unc = cv.imread(path)
            thresh = masking(photo_unc,px_too_dark,px_too_bright)[3]	
            
            photo_unc = cv.resize(photo_unc, (0, 0), None, .20, .20)
            thresh = cv.resize(thresh, (0, 0), None, .20, .20)
            thresh_3_channel = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
            numpy_horizontal = np.hstack((photo_unc, thresh_3_channel))
            cv.imshow('Press -y- to accept this photo, else -n-', numpy_horizontal)
            key = cv.waitKey(0)
            if key == ord('y'):
                    print ('ok')
                    df_results.at[i,'suitable?'] = 'Y'
                    cv.destroyWindow('Press -y- to accept this photo, else -n-')
            elif key == ord('n'):
                    print ('not ok')
                    df_results.at[i,'suitable?'] = 'N'
                    cv.destroyWindow('Press -y- to accept this photo, else -n-')
                    
        return df_results
    
    
#### EXTRACTION OF EXIF DATA FROM PHOTOS ####

def EXIF_extract(photo):
    
        return Image.open(photo)._getexif()[36867]

#### MANIPULATION OF TXT FILES FOR CAMERA ALTITUDE ####


def cam_altitude(txt_src):
    
        print ('Extracting data from the video profile raw text files \n')
        df = pd.DataFrame()
        
        for dirs, subdirs, files in os.walk(txt_src):
                for f in fnmatch.filter(files,'*.txt'):
                        print ('Working on ' +f)
                        dftemp = pd.read_table(os.path.join(dirs, f), delimiter=',', encoding='latin1', 
                                               names=('Datetime', 'Serial', 'Data'))
                        
                        dftemp['Datetime'] = dftemp['Datetime'].str.replace('SerA:',',SerA:,',1).str.replace('SerB:',',SerB:,',1).str.replace('SerC:',',SerC:,',1)
                        dftemp[['Datetime', 'Serial', 'Data']] =dftemp['Datetime'].str.split(',',expand =True)

                        dftemp['Datetime'] = pd.to_datetime(dftemp['Datetime']).values.astype('datetime64[s]')#values.astype('datetime64[s]')

                        dftemp['Depth(m)'] = dftemp['Data'].str.extract(r'Depth: (.*)m')
                        dftemp['Lat'] = dftemp['Data'].str.extract(r'Lat:(.*)Lon:')
                        dftemp['Lon'] = dftemp['Data'].str.extract(r'Lon:(.*)')
                        dftemp.Lat = dftemp.Lat.str.replace(' ', '')
                        dftemp.Lon = dftemp.Lon.str.replace(' ', '')
                        dftemp = dftemp[~dftemp['Lat'].str.contains('-', na=False)]
                        dftemp['Lat'] = dftemp['Lat'].apply(lambda x: dms2dd(x) if pd.notnull(x) else x)
                        dftemp['Lon'] = dftemp['Lon'].apply(lambda x: dms2dd(x) if pd.notnull(x) else x)

                        dftemp['Altitude'] = dftemp['Data'].str.extract(r'Alt:(.*)Temp:')
                        dftemp['Temp'] = dftemp['Data'].str.extract(r'Temp:(.*)')
                        dftemp['Bearing'] = dftemp['Data'].str.extract(r'Brg:(.*)Depth:')
                        
                        df = pd.concat([df, dftemp])
        
        print ('extraction completed')

        df2 = df.drop(columns=['Serial', 'Data'])
        df2[['Depth(m)','Lat','Lon','Altitude','Temp', 'Bearing']] = df2[['Depth(m)','Lat','Lon','Altitude','Temp', 'Bearing']].apply(pd.to_numeric, errors='coerce')
        df2 = df2.groupby(df2['Datetime']).mean()
        df2 = df2.reset_index()
        
        return df2

from PIL import Image, ExifTags

def rotateImg (path):
    img_pil = Image.open(path)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
             break
    
    exif = img_pil._getexif()
    
    if exif[orientation] == 3:
        img_pil=img_pil.rotate(180, expand=True)
    elif exif[orientation] == 6:
        img_pil=img_pil.rotate(270, expand=True)
    elif exif[orientation] == 8:
        img_pil=img_pil.rotate(90, expand=True)

    img_pil.save(path, format='JPEG', exif=img_pil.info['exif'])
    img_pil.close()
    pass

### GENERAL PROCESS FUNCTIONS ###


def altimetry_blur_filter(src, dst_tables):
    
        start = time.time()    
                        
        df_EXIF_DT = pd.DataFrame(columns =['Photo_path', 'Photo_ID', 'Datetime', 'blur_value']) 
            
        with open(os.path.join(dst_tables, 'blur_EXIF.csv'),'w') as f1:
                
                writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
                
                writer.writerow(['Photo_ID', 'blur_value', 'datetime'])
    
                for src, subdir, pho in os.walk(src):
                        for p in fnmatch.filter(pho,'*.jpg'):
                                print ('Now processing ' + p)
                                path = os.path.join(src, p)
                                rotateImg(path) #reads the exif and rotates based on the camera value        
                                photo = cv.imread(path)
                                blur = variance_of_laplacian(photo)
            
                                datetime = EXIF_extract(path)
            
                                df_EXIF_DT = df_EXIF_DT.append({'Photo_path':path, 'Photo_ID':p, 'Datetime':datetime, 'blur_value':blur}, ignore_index=True)
                                
                                writer.writerow([p, blur, datetime])
                                
                finish = str((time.time() - start)/60)[0:4]
                print("Processed {} images in {} minutes".format(len(df_EXIF_DT['Photo_ID']),finish))      
        
        return df_EXIF_DT


def photo_quality_assess(src, dst, dst_tables, perc_unusable_img, px_too_dark, px_too_bright, sk_up, sk_low, ku_up):

        start = time.time()  
        suitable = 0
        
        #### final dataframe with results ####

        df_results = pd.DataFrame(columns =['Photo_ID', 'skew', 'kurtosis', 'dark pixels', 'bright pixels','unusable_image_%', 
                                            'n_unusable_patches', 'largest_unusable_patch_%', 'IQA score', 'suitable?'])
        
        with open(os.path.join(dst_tables, 'Quality_Assess_selection.csv'),'w') as f1:
            
                writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
                
                writer.writerow(['Photo_ID', 'skew', 'kurtosis', 'dark pixels', 'bright pixels',
                                'unusable_image_%', 'n_unusable_patches', 'largest_unusable_patch_%', 
                                'suitable?'])
                
                for src, subdir, pho in os.walk(src):
                        for p in fnmatch.filter(pho,'*.jpg'):
                                print ('Now processing ' + p)
                                path = os.path.join(src, p)
                                photo = cv.imread(path)
                               
                                _, _, _, a_skw, a_kurt, b_dark_pixels, b_bright_pixels = histograms(photo, px_too_dark, px_too_bright)
        
                                whitepix_perc, blobcount, Max_blob, thresh = masking(photo, px_too_dark, px_too_bright)
                                IQA_score = IQA_brisque(path)        
            
                                if whitepix_perc >= perc_unusable_img:
                                        suitable = 'N'
                                elif whitepix_perc < perc_unusable_img and a_skw > sk_up:
                                        suitable = 'Uncertain'
                                elif whitepix_perc < perc_unusable_img and a_skw < sk_low:
                                        suitable = 'Uncertain'
                                elif whitepix_perc < perc_unusable_img and a_kurt > ku_up:
                                        suitable = 'Uncertain'
                                else:
                                        suitable = 'Y'

            
                                df_results = df_results.append({'Photo_ID':p, 'skew':a_skw, 'kurtosis':a_kurt, 'dark pixels':b_dark_pixels, 'bright pixels':b_bright_pixels,
                                                                'unusable_image_%':whitepix_perc, 'n_unusable_patches':blobcount, 'largest_unusable_patch_%':Max_blob, 
                                                                'IQA score': IQA_score,'suitable?':suitable}, ignore_index=True)
                
                                writer.writerow([p, a_skw, a_kurt, b_dark_pixels, b_bright_pixels, whitepix_perc, blobcount, Max_blob, suitable])
                
                                if suitable == 'Uncertain':
                                        copyFile(path, os.path.join(dst, p), buffer_size=10485760, perserveFileDate=True)

                          
        finish = str((time.time() - start)/60)[0:4]

        print("Processed {} images in {} minutes".format((df_results['Photo_ID']),finish))        
        
        return df_results
    
    
    
    #############################################
        ######################################
################### BRISQUE IQA #########################
        ######################################
    #############################################
    
### code taken from Ricardo Ocasio's notebook
#https://github.com/ocampor/notebooks/blob/master/notebooks/image/quality/brisque.ipynb
        
import collections
from itertools import chain
import pickle 

import scipy.signal as signal
import scipy.special as special
import scipy.optimize as optimize

import skimage.io
import skimage.transform

from libsvm import svmutil


def normalize_kernel(kernel):
        return kernel / np.sum(kernel)

def gaussian_kernel2d(n, sigma):
        Y, X = np.indices((n, n)) - int(n/2)
        gaussian_kernel = 1 / (2 * np.pi * sigma ** 2) * np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) 
        return normalize_kernel(gaussian_kernel)

def local_mean(image, kernel):
        return signal.convolve2d(image, kernel, 'same')

def local_deviation(image, local_mean, kernel):
        "Vectorized approximation of local deviation"
        sigma = image ** 2
        sigma = signal.convolve2d(sigma, kernel, 'same')
        return np.sqrt(np.abs(local_mean ** 2 - sigma))

def calculate_mscn_coefficients(image, kernel_size=6, sigma=7/6):
        C = 1/255
        kernel = gaussian_kernel2d(kernel_size, sigma=sigma)
        local_mean = signal.convolve2d(image, kernel, 'same')
        local_var = local_deviation(image, local_mean, kernel)
        
        return (image - local_mean) / (local_var + C)

def generalized_gaussian_dist(x, alpha, sigma):
        beta = sigma * np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        
        coefficient = alpha / (2 * beta() * special.gamma(1 / alpha))
        return coefficient * np.exp(-(np.abs(x) / beta) ** alpha)

def calculate_pair_product_coefficients(mscn_coefficients):
        return collections.OrderedDict({
            'mscn': mscn_coefficients,
            'horizontal': mscn_coefficients[:, :-1] * mscn_coefficients[:, 1:],
            'vertical': mscn_coefficients[:-1, :] * mscn_coefficients[1:, :],
            'main_diagonal': mscn_coefficients[:-1, :-1] * mscn_coefficients[1:, 1:],
            'secondary_diagonal': mscn_coefficients[1:, :-1] * mscn_coefficients[:-1, 1:]
        })

def asymmetric_generalized_gaussian(x, nu, sigma_l, sigma_r):
        def beta(sigma):
                return sigma * np.sqrt(special.gamma(1 / nu) / special.gamma(3 / nu))
        
        coefficient = nu / ((beta(sigma_l) + beta(sigma_r)) * special.gamma(1 / nu))
        f = lambda x, sigma: coefficient * np.exp(-(x / beta(sigma)) ** nu)
            
        return np.where(x < 0, f(-x, sigma_l), f(x, sigma_r))

def asymmetric_generalized_gaussian_fit(x):
        def estimate_phi(alpha):
                numerator = special.gamma(2 / alpha) ** 2
                denominator = special.gamma(1 / alpha) * special.gamma(3 / alpha)
                return numerator / denominator

        def estimate_r_hat(x):
                size = np.prod(x.shape)
                return (np.sum(np.abs(x)) / size) ** 2 / (np.sum(x ** 2) / size)

        def estimate_R_hat(r_hat, gamma):
                numerator = (gamma ** 3 + 1) * (gamma + 1)
                denominator = (gamma ** 2 + 1) ** 2
                return r_hat * numerator / denominator

        def mean_squares_sum(x, filter = lambda z: z == z):
                filtered_values = x[filter(x)]
                squares_sum = np.sum(filtered_values ** 2)
                return squares_sum / ((filtered_values.shape))
    
        def estimate_gamma(x):
                left_squares = mean_squares_sum(x, lambda z: z < 0)
                right_squares = mean_squares_sum(x, lambda z: z >= 0)
        
                return np.sqrt(left_squares) / np.sqrt(right_squares)
    
        def estimate_alpha(x):
                r_hat = estimate_r_hat(x)
                gamma = estimate_gamma(x)
                R_hat = estimate_R_hat(r_hat, gamma)
        
                solution = optimize.root(lambda z: estimate_phi(z) - R_hat, [0.2]).x
        
                return solution[0]
    
        def estimate_sigma(x, alpha, filter = lambda z: z < 0):
                return np.sqrt(mean_squares_sum(x, filter))
        
        def estimate_mean(alpha, sigma_l, sigma_r):
                return (sigma_r - sigma_l) * constant * (special.gamma(2 / alpha) / special.gamma(1 / alpha))
        
        alpha = estimate_alpha(x)
        sigma_l = estimate_sigma(x, alpha, lambda z: z < 0)
        sigma_r = estimate_sigma(x, alpha, lambda z: z >= 0)
        
        constant = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
        mean = estimate_mean(alpha, sigma_l, sigma_r)
        
        return alpha, mean, sigma_l, sigma_r

def calculate_brisque_features(image, kernel_size=7, sigma=7/6):
        def calculate_features(coefficients_name, coefficients, accum=np.array([])):
                alpha, mean, sigma_l, sigma_r = asymmetric_generalized_gaussian_fit(coefficients)
    
                if coefficients_name == 'mscn':
                        var = (sigma_l ** 2 + sigma_r ** 2) / 2
                        return [alpha, var]
            
                return [alpha, mean, sigma_l ** 2, sigma_r ** 2]
    
        mscn_coefficients = calculate_mscn_coefficients(image, kernel_size, sigma)
        coefficients = calculate_pair_product_coefficients(mscn_coefficients)
        
        features = [calculate_features(name, coeff) for name, coeff in coefficients.items()]
        flatten_features = list(chain.from_iterable(features))
        return np.array(flatten_features)

def scale_features(features):
        with open('normalize.pickle', 'rb') as handle:
                scale_params = pickle.load(handle)
        
        min_ = np.array(scale_params['min_'])
        max_ = np.array(scale_params['max_'])
        
        return -1 + (2.0 / (max_ - min_) * (features - min_))

def calculate_image_quality_score(brisque_features):
        model = svmutil.svm_load_model('brisque_svm.txt')
        scaled_brisque_features = scale_features(brisque_features)
        
        x, idx = svmutil.gen_svm_nodearray(
                scaled_brisque_features,
                isKernel=(model.param.kernel_type == svmutil.PRECOMPUTED))
        
        nr_classifier = 1
        prob_estimates = (svmutil.c_double * nr_classifier)()
        
        return svmutil.libsvm.svm_predict_probability(model, x, prob_estimates)

def plot_histogram(x, label):
        n, bins = np.histogram(x.ravel(), bins=50)
        n = n / np.max(n)
        plt.plot(bins[:-1], n, label=label, marker='o')
        
####main Brisque function####
def IQA_brisque(i):
        plt.rcParams["figure.figsize"] = 12, 9
        
        image = skimage.io.imread(i, plugin='pil')
        gray_image = skimage.color.rgb2gray(image)
        
        _ = skimage.io.imshow(image)
        
        mscn_coefficients = calculate_mscn_coefficients(gray_image, 7, 7/6)
        coefficients = calculate_pair_product_coefficients(mscn_coefficients)
        plt.rcParams["figure.figsize"] = 12, 11
        
        for name, coeff in coefficients.items():
                plot_histogram(coeff.ravel(), name)
        
        plt.axis([-2.5, 2.5, 0, 1.05])
        plt.legend()
        plt.show()
        
        brisque_features = calculate_brisque_features(gray_image, kernel_size=7, sigma=7/6)
        
        downscaled_image = cv.resize(gray_image, None, fx=1/2, fy=1/2, interpolation = cv.INTER_CUBIC)
        downscale_brisque_features = calculate_brisque_features(downscaled_image, kernel_size=7, sigma=7/6)
        
        brisque_features = np.concatenate((brisque_features, downscale_brisque_features))
        
        IQA_score = calculate_image_quality_score(brisque_features)
        
        return IQA_score