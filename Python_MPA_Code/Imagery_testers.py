# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 08:33:00 2020

@author: RA05
"""

import Photo_suitability_functions as ps

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import configparser

#*****************************INITIAL TESTS*****************************

config = configparser.ConfigParser()
config.read('Image_thresholds.INI')

test_src = config['Folders']['test_src']

#list the lower and upper pixel value brightness thresholds you want to test, a couplet will be formed following the order
#e.g. if list_low = [30,40,50] and list_up = [220,200,190] then the couplets will be: 30:220, 40:200 and 50:190

list_low = [30, 40, 50]
list_up = [220, 200, 190]

ps.parameter_tester(test_src, list_low, list_up)



            
