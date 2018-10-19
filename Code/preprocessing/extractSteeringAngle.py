# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 14:36:19 2018

Extrahiert Bildnummer und Lenkwinkel aus einmel Ordner mit Bildern
(formatiert nach automatischer Benennung durch Software auf Carolo-Cup-Fahrzeug)
und schreibt diese als Tupel in eine Textdatei

@author: RR
"""


import cv2
import numpy as np
import os, os.path

imageDir = "C:/Users/user/Desktop/BA/Bilder/clean cropped data betterVersion"
image_path_list = []

for file in os.listdir(imageDir):
    image_path_list.append(os.path.join(imageDir, fle))
    
tupel =[]
for imagePath in image_path_list:
    filename = imagePath
    #image = cv2.imread(filename,0)
    #print(filename)
    SplittedFilename = filename.split('_')
    #print(filename.split('_'))
    
    tupel.append("%s|||%s" % (SplittedFilename[1],SplittedFilename[3]))
                            #Bildnummer|||Lenkwinkel
                            
                            
AlleTupel = "\n".join(tupel)
#print(AlleTupel)
f = open("C:/Users/user/Desktop/BA/Bilder/clean cropped data betterVersion/Lenkwinkel.txt", "w")
f.write(AlleTupel)
f.close() 