import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.stats import norm, multivariate_normal
from numpy.random import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import os
import time as t
import csv

def multinomial_resample(weights):

    weights=weights.T
    cumulative_sum = np.cumsum(weights)
    #cumulative_sum[-1] = 1.  # avoid round-off errors: ensures sum is exactly one
    #print ( np.searchsorted(cumulative_sum, random(len(weights))))
    return np.searchsorted(cumulative_sum, random(len(weights)))


def lecture_image() :

    SEQUENCE = "./sequences/sequence1/sequence1/"
    #charge le nom des images de la séquence
    filenames = os.listdir(SEQUENCE)
    T = len(filenames)
    #charge la premiere image dans ’im’
    tt = 0

    im=Image.open((str(SEQUENCE)+str(filenames[tt])))
    plt.imshow(im)

    return(im,filenames,T,SEQUENCE)

def selectionner_zone() :

    #lecture_image()
    print('Cliquer 4 points dans l image pour definir la zone a suivre.') ;
    zone = np.zeros([2,4])
 #   print(zone))
    compteur=0
    while(compteur != 4):
        res = plt.ginput(1)
        a=res[0]
        #print(type(a)))
        zone[0,compteur] = a[0]
        zone[1,compteur] = a[1]
        plt.plot(a[0],a[1],marker='X',color='red')
        compteur = compteur+1

    #print(zone)
    newzone = np.zeros([2,4])
    newzone[0, :] = np.sort(zone[0, :])
    newzone[1, :] = np.sort(zone[1, :])

    zoneAT = np.zeros([4])
    zoneAT[0] = newzone[0,0]
    zoneAT[1] = newzone[1,0]
    zoneAT[2] = newzone[0,3]-newzone[0,0]
    zoneAT[3] = newzone[1,3]-newzone[1,0]
    #affichage du rectangle
    #print(zoneAT)
    xy=(zoneAT[0],zoneAT[1])
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None')
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    plt.show(block=False)
    return(zoneAT)


def rgb2ind(im,nb) :
    #nb = nombre de couleurs ou kmeans qui contient la carte de couleur de l'image de référence
    print(im)
    image=np.array(im,dtype=np.float64)/255
    w,h,d=original_shape=tuple(image.shape)
    image_array=np.reshape(image,(w*h,d))
    image_array_sample=shuffle(image_array,random_state=0)[:1000]
    print(image_array_sample.shape)
   # print(type(image_array))
    if type(nb)==int :
        kmeans=KMeans(n_clusters=nb,random_state=0).fit(image_array_sample)
    else :
        kmeans=nb

    labels=kmeans.predict(image_array)
    #print(labels)
    image=recreate_image(kmeans.cluster_centers_,labels,w,h)
    #print(image)
    return(Image.fromarray(image.astype('uint8')),kmeans)

def recreate_image(codebook,labels,w,h):
    d=codebook.shape[1]
    #image=np.zeros((w,h,d))
    image=np.zeros((w,h))
    label_idx=0
    for i in range(w):
        for j in range(h):
            #image[i][j]=codebook[labels[label_idx]]*255
            image[i][j]=labels[label_idx]
            #print(image[i][j])
            label_idx+=1

    return image



def calcul_histogramme(im,zoneAT,Nb):

  #  print(zoneAT)
    box=(zoneAT[0],zoneAT[1],zoneAT[0]+zoneAT[2],zoneAT[1]+zoneAT[3])
   # print(box)
    littleim = im.crop(box)
##    plt.imshow(littleim)
##    plt.show()
    new_im,kmeans= rgb2ind(littleim,Nb)
    histogramme=np.asarray(new_im.histogram())
    #print(histogramme)
##  print(histogramme)
    histogramme=histogramme/np.sum(histogramme)
  #  print(new_im)
    #plt.imshow(new_im)
    #plt.show()
    #wait = input("Press Enter to continue.")
    #plt.close()
    return (new_im,kmeans,histogramme)


N=100
Nb=20
ecart_type=np.sqrt(50)
lambda_im=60
c1=pow(300,1/2)
c2=pow(300,1/2)
c3 = 20/100
V=np.diag([c1,c2])
C=np.diag([c1,c2,c3])

[im,filenames,T,SEQUENCE]=lecture_image()
zoneAT=selectionner_zone()


new_im,kmeans,histo_ref=calcul_histogramme(im,zoneAT,Nb)

def calcul_de_vraissemblance(histo, histo_ref,zone):
    if zone[0] > 500 or zone[0] < - zone[2] or zone[1] > 500 or zone[1] < - zone[3]:
        return(0)

    return np.exp(-lambda_im*(1-((histo*histo_ref)**(1/2)).sum()))



def filtrage_particulaire():
    X_0 = np.array(np.random.randn(N,2))*c1 + [zoneAT[0],zoneAT[1]]
    w = []
    for i in range(N):
        w += [multivariate_normal.pdf(np.array([zoneAT[0],zoneAT[1]]),X_0[i],V)]
    w = np.array(w)
    w = w/w.sum()

    x_est = np.zeros((T-1,3))
    x = np.zeros((T,N,3))
    x[0,:,0:2] = X_0
    x[0,:,2] += np.ones(N)
    for t in range(1,T):
        im = Image.open((str(SEQUENCE)+str(filenames[t])))
        A = np.random.choice(range(N),N,p=w)
        reech = x[t-1,:][A] #C'est l'étape de réechantillonnage
        #On détermine désormais les nouvelles particules en utilisant la fonction de transition
        x[t,:] = reech
        x[t,:,0:2] += np.array(np.random.randn(N,2))*c1
        x[t,:,2] += np.array(np.random.randn(N))*c3
        #on calcule les poids de chaque particule en utilisant leur vraissemblance
        w = [calcul_de_vraissemblance(histo_ref,calcul_histogramme(im,[x[t,i,0],x[t,i,1],zoneAT[2]*x[t,i,2],zoneAT[3]*x[t,i,2]],kmeans)[2],[x[t,i,0],x[t,i,1],zoneAT[2]*x[t,i,2],zoneAT[3]*x[t,i,2]]) for i in range(N)]#Le paramètre d'échelle intervient à cet endroit
        w = np.array(w)
        w=w/w.sum() #C'est le vecteur des poids normalisés de chaque particules

        #On construit l'estimation t à partir des N particules et de leurs poids relatifs
        for i in range(N):
            x_est[t-1,2] += x[t,i,2]*w[i]
            x_est[t-1,0:2] += x[t,i,0:2]*w[i]

    return(x_est,x)

x_est,x = filtrage_particulaire() #x_est est la trajectoire estimée

wait = input("Press Enter to continue.") 
plt.close()

#On va maintenant afficher les images une par une
for t in range(1,T):
    im = Image.open((str(SEQUENCE)+str(filenames[t])))
    plt.imshow(im)
    #On trace un rectangle rouge correspondant à l'estimation sur cette images
    xy = x_est[t-1]
    rect=ptch.Rectangle(xy,zoneAT[2],zoneAT[3],linewidth=3,edgecolor='red',facecolor='None')
    #plt.Rectangle(zoneAT[0:1],zoneAT[2],zoneAT[3])
    currentAxis = plt.gca()
    currentAxis.add_patch(rect)
    #On affiche également en vert toutes les particules qui ont permis de déterminer l'estimation
    for i in range(N):
        plt.plot(z[t,i,0],z[t,i,1],marker='X',color='green')

    plt.show()
    wait = input("Press Enter to continue.")
    plt.close()
