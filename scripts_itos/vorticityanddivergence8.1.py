#!/usr/bin/env python
"""
    vorticity and divergence tlat 8.1.py
    @author Pedraza Espitia S.
    @version 1.0
"""
# 
import numpy as np
import cv2
import matplotlib.pyplot as plt  


seriemaximosvort = []
serieminimosvort = []
seriemaximosdive = []
serieminimosdive = []

cap = cv2.VideoCapture("MVI_2606.MOV")
t_seg = 3*60+10
cap.set(cv2.CAP_PROP_POS_MSEC,t_seg*1000)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

x = np.arange(np.shape(prvs)[1]) # num de cols
y = np.arange(np.shape(prvs)[0]) # num de fils
X,Y = np.meshgrid(x,y)
# (1920cols,1080rows)

ventanayfilsi = 100 # inicio (esquina de ventana inf izq)
ventanaxcolsi = 400 # inicio (esquina de ventana inf izq)
ventanayfils = 800
ventanaxcols = 1400 # tamanio de ventana

print('running')

# plt.contourf(prvs[::-1])
# plt.show()

for i in range(1,401):
	ret, frame2 = cap.read()
	siguiente = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(prvs[::-1],siguiente[::-1], None, pyr_scale=0.5,
	                                    levels=1,
	                                    winsize=25,
	                                    iterations=1,
	                                    poly_n=5,
	                                    poly_sigma=1.1,
	                                    flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	U = flow[...,0]
	V = flow[...,1]
	dosvort = np.zeros((ventanayfils-2, ventanaxcols-2))
	dosdive = np.zeros((ventanayfils-2, ventanaxcols-2))
	U = U[ventanayfilsi:ventanayfilsi+ventanayfils,ventanaxcolsi:ventanaxcolsi+ventanaxcols]
	V = V[ventanayfilsi:ventanayfilsi+ventanayfils,ventanaxcolsi:ventanaxcolsi+ventanaxcols]

	for rowcoory in range(1,ventanayfils-1):
		for colcoorx in range(1,ventanaxcols-1):
			dosdvdx = V[rowcoory,colcoorx+1] - V[rowcoory,colcoorx-1]
			dosdudy = U[rowcoory+1,colcoorx] - U[rowcoory-1,colcoorx]
			dosdvdy = V[rowcoory+1,colcoorx] - V[rowcoory-1,colcoorx]
			dosdudx = U[rowcoory,colcoorx+1] - U[rowcoory,colcoorx-1]
			dosvort[rowcoory-1,colcoorx-1] = (dosdvdx - dosdudy)#/2 comentado para escalar 2x la vort
			dosdive[rowcoory-1,colcoorx-1] = (dosdudx + dosdvdy)#/2
	


	prvs = prvs[::-1]
	prvs.tofile("lava/lavanp"+str('%03d' % i)+"im.dat")
	U.tofile("lava/lavanp"+str('%03d' % i)+"U.dat")
	V.tofile("lava/lavanp"+str('%03d' % i)+"V.dat")
	dosvort.tofile("lava/lavanp"+str('%03d' % i)+"dosvort.dat")
	dosdive.tofile("lava/lavanp"+str('%03d' % i)+"dosdive.dat")
	print(i)
	prvs = siguiente
cap.release()
cv2.destroyAllWindows()
