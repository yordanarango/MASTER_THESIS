# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 04:54:14 2016

@author: yordan
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd
from netcdftime import utime
import netCDF4 as nc



#
'DATOS CCMP'
#
TT      = []
PP      = []
PN      = []

Dates = pd.date_range('1998-01-01', '2010-12-08', freq='D')
for date in Dates:
	Y       = str(date.year)
	M       = '%02d' % (date.month,)
	D       = '%02d' % (date.day,)
	archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/'+Y+'/CCMP_Wind_Analysis_'+Y+M+D+'_V02.0_L3.0_RSS.nc')

	U_TT    = archivo.variables['uwnd'][:, 364:379, 1055:1066]; V_TT    = archivo.variables['vwnd'][:, 364:379, 1055:1066]
	U_PP    = archivo.variables['uwnd'][:, 349:372, 1078:1097]; V_PP    = archivo.variables['vwnd'][:, 349:372, 1078:1097]
	U_PN    = archivo.variables['uwnd'][:, 336:349, 1117:1126]; V_PN    = archivo.variables['vwnd'][:, 336:349, 1117:1126]
	
	spd_TT = np.sqrt(U_TT*U_TT + V_TT*V_TT)
	spd_PP = np.sqrt(U_PP*U_PP + V_PP*V_PP)
	spd_PN = np.sqrt(U_PN*U_PN + V_PN*V_PN)
	
	for i in range(4):
		TT.append(np.mean(spd_TT[i]))
		PP.append(np.mean(spd_PP[i]))
		PN.append(np.mean(spd_PN[i]))

TT = np.array(TT)
PP = np.array(PP)
PN = np.array(PN)

#
'DATOS RASI'
#

RASI_Tehuantepec = []
RASI_Papagayo    = []
RASI_Panama      = []

for i in range(1998, 2012):
	
	mean_TT = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Tehuantepec_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PP = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Papagayo_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PN = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Panama_'+str(i)+'.nc')['WindSpeedMean'][:]

	RASI_Tehuantepec.extend(mean_TT)
	RASI_Papagayo.extend(mean_PP)
	RASI_Panama.extend(mean_PN)

RASI_Tehuantepec = np.array(RASI_Tehuantepec)[:len(Dates)*4]
RASI_Papagayo    = np.array(RASI_Papagayo)[:len(Dates)*4]
RASI_Panama      = np.array(RASI_Panama)[:len(Dates)*4]


# cdftime = utime('seconds since 2010-01-01T00:00:00Z', calendar='gregorian')
# date    = [cdftime.num2date(k) for k in prueba_time]
# date    = pd.DatetimeIndex(date)
 


#
'FUNCIÓN INDICES'
#

def indices(Serie): # Serie que contiene dato cuando hay evento de chorro, y ceros cuando no se considera que hay evento de chorro
	EVENTOS = []
	i = 0
	while i < len(Serie)-1:

		if Serie[i] != 0:
			aux = i
			while Serie[aux+1] != 0:
				aux = aux + 1
				if aux == len(Serie)-1:
					break
			fin = aux

			EVENTOS.append([i, fin])

			i = aux + 2 # Siguiente posicion a examinar
		else:
			i = i + 1 # Siguiente posicion a examinar

	return np.array(EVENTOS)

EVN_TT = indices(RASI_Tehuantepec)
EVN_PP = indices(RASI_Papagayo)
EVN_PN = indices(RASI_Panama)

#
'FUNCION GRAFICAR DERIVADAS'
#

def derv_plt(EVN, spd_CCMP, path, m):
	#EVN      : array de numpy que contiene pares que indican comieno y fin de eventos presentes en datos de RASI
	#spd_CCMP : array con composites de velocidades para el chorro en cuestión, derivado de datos de CCMP
	#path     : string del directorio donde se guardará la imagen
	#m        : valor umbral de selección de los eventos. Para este caso se aplica al principio de cada evento, es decir, 36 horas antes del climax del evento

	fig = plt.figure(figsize=(11,5))
	ax1 = fig.add_subplot(111)
	for i in range(len(EVN[:,0])):
		if EVN[i,0] != EVN[i,1]:
			print i
			c = EVN[i,0]+np.where(spd_CCMP[EVN[i,0]:EVN[i,1]+1] == np.max(spd_CCMP[EVN[i,0]:EVN[i,1]+1]))
			cc = c[0,0]
			if cc <= len(spd_CCMP)-7:        
				if m <= spd_CCMP[cc-6] < m+1:
					der = np.zeros(len(spd_CCMP[cc-6:cc+7]))                
					for j in range(len(der)):
						#print cc-6+j+1, cc-6+j
						der[j] = spd_CCMP[cc-6+j+1]-spd_CCMP[cc-6+j]
					ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12) ,der, linewidth=0.7, color='k', alpha = 0.7)
		else:
			if EVN[i,1] <= len(spd_CCMP)-7:        
				if m <= spd_CCMP[EVN[i,1]-6]< m+1:
					der = np.zeros(13)
					for j in range(len(der)):
						der[j] = spd_CCMP[EVN[i,1]-6+j+1]-spd_CCMP[EVN[i,1]-6+j] 
					ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12), der, linewidth=0.7, color='k', alpha = 0.7)
	my_xticks = ['-36:00','-30:00','-24:00', '-18:00', '-12:00', '-06:00', '00:00', '06:00', '12:00', '18:00', '24:00', '30:00', '36:00']
	plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),my_xticks)
	ax1.grid(True)
	ax1.set_xlabel('$Time$ $(h)$', size='15')
	ax1.set_ylabel('$\Delta$ $Speed$ $(m/s)$', size='15')
	ax1.set_title('Wind Speed Derived, '+str(float(m))+' - '+str(float(m+1))+' (TT)', size='14')
	ax1.legend(loc='best')
	ax1.set_ylim(-5.5, 5.5)
	plt.savefig(path+str(m)+'-'+str(m+1)+'.png',dpi=100,bbox_inches='tight')

#
'GRAFICANDO'
#
PATH = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/4_expo_2017/DERIVADAS_CCMP/COMENZANDO/PN/'
for i in range(20):
	derv_plt(EVN_PN[2:], PN,  PATH, i)










archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/2005/CCMP_Wind_Analysis_20050102_V02.0_L3.0_RSS.nc')




