# -*- coding: utf-8 -*-
"""
Created on Sat May 15 13:37:10 2021

@author: Natsy
"""

import cv2 as cv

# charger les classificateurs en cascade pré-entrainés

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# charger les images 

img = cv.imread('obama.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# exécution de la detection des visages
# detectMultiScale(image, paramettre echelle, nombre de voisins)

faces = face_cascade.detectMultiScale(gray, 1.1, 8)

# affichage des visages

i = 0
for face in faces:
    x, y, w, h = face
    
    #dessiner le rectangle sur l'image
    #rectangle(image, coordonné haut gauche visage, coordonné bas droite visage, couleur rectangle, epaisseur rectangle)
    
    cv.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)
    
    # extraire les visages de l'image principale
    # OpenCV et Numpy: y <-> row et x <-> column
    
    face = img[y:y+h, x:x+w]
    
    #afficher face0, face1 ...
    cv.imshow('face{}'.format(i), face) 
    i += 1
    
#affiche l'image principale
cv.imshow('image principale', img)
cv.waitKey(0)
cv.destroyAllWindows()