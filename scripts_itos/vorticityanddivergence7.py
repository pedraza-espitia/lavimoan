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
	

	fig = plt.figure(1)
	ax1 = fig.add_subplot(121)
	plt.figure(figsize=(17,9))
	prvs = prvs[::-1]
	plt.imshow(prvs[ventanayfilsi:ventanayfilsi+ventanayfils,ventanaxcolsi:ventanaxcolsi+ventanaxcols])
	plt.quiver(X[0:ventanayfils:20,0:ventanaxcols:20],
	           Y[0:ventanayfils:20,0:ventanaxcols:20],U[::20,::20],V[::20,::20])
	plt.gca().invert_yaxis()
	
	plt.savefig("lava/lava"+str('%03d' % i)+".png", dpi = 200, bbox_inches='tight',
		pad_inches=0);plt.close(1)
	#plt.show(1); # '{:02d}'.format(i)

	fig = plt.figure(2)
	ax2 = fig.add_subplot(111)
	cf2 = ax2.contourf(dosdive, np.arange(-3.5, 4.3, .4),
	                    extend='both')
	cbar2 = plt.colorbar(cf2)
	plt.quiver(X[0:ventanayfils:20,0:ventanaxcols:20],
	           Y[0:ventanayfils:20,0:ventanaxcols:20],U[::20,::20],V[::20,::20])
	seriemaximosdive.append(np.max(dosdive))
	serieminimosdive.append(np.min(dosdive))
	plt.savefig("dive/dive"+str('%03d' % i)+".png", dpi = 200, bbox_inches='tight',\
		pad_inches=0);plt.close(2) # '{:02d}'.format(i)
	# #plt.show(2);#

	fig = plt.figure(3)
	ax3 = fig.add_subplot(111)
	cf3= plt.contourf(dosvort,np.arange(-3.5, 3, .4),
	                   extend='both')
	cbar3 = plt.colorbar(cf3)
	plt.quiver(X[0:ventanayfils:20,0:ventanaxcols:20],
	           Y[0:ventanayfils:20,0:ventanaxcols:20],U[::20,::20],V[::20,::20])
	seriemaximosvort.append(np.max(dosvort))
	serieminimosvort.append(np.min(dosvort))
	plt.savefig("vort/vort"+str('%03d' % i)+".png", dpi = 200, bbox_inches='tight',\
		pad_inches=0);plt.close(3) #plt.show() #plt.clf()
	plt.show(3)
	print(i)
	prvs = siguiente
cap.release()
cv2.destroyAllWindows()
