# -*- coding: utf-8 -*-
"""
Created on Sun May 16 22:46:24 2021

@author: Natsy
"""
import cv2 as cv
import sys

# charger les classificateurs en cascade pré-entrainés

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images 

img = cv.imread('brad-angelina.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# exécution de la detection des visages
# detectMultiScale(image, paramettre echelle, nombre de voisins)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)

#vérifier le nombre de visages
if len(faces) != 2:
    sys.exit('La photo doit avoir 2 visages')

# récupération des dimensions de chaque visage
x1, y1, w1, h1 = faces[0]
x2, y2, w2, h2 = faces[1]

#extractions des 2 visages de l'image
face1 = img[y1:y1+h1, x1:x1+w1]
face2 = img[y2:y2+h2, x2:x2+w2]

#redimensionner face2 aux dimensions face1 et vice versa
face2 = cv.resize(face2, (w1, h1))
face1 = cv.resize(face1, (w2, h2))

#remplacer face2 par face1
img[y2:y2+h2, x2:x2+w2] = face1

#remplacer face1 par face2
img[y1:y1+h1, x1:x1+w1] = face2
 
#afficher l'echange de visage

cv.imshow('echange', img)
cv.waitKey(0)
cv.destroyAllWindows()



