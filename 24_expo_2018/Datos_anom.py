# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:34:25 2018

@author: yordan
"""
import numpy as np
import netCDF4 as nc
from netcdftime import utime
import pandas as pd
from pandas import Timestamp
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import csv

"Escoger chorro: TT, PP, PN"
ch = 'PN'

"Leyendo datos"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area_925hPa.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-360

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas) # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas

"Fecha hasta donde se va a hacer HMM"
pos_2017_12_31 = np.where(DATES == Timestamp('2017-12-31 18:00:00'))[0][0]
DATES          = DATES[3 : pos_2017_12_31+1 : 4]
TIME           =  time[3 : pos_2017_12_31+1 : 4]

"coordenadas"
if ch == 'TT':
	lat_ch = 15  ; lon_ch = -95
elif ch == 'PN':
	lat_ch = 7   ; lon_ch = -79.5
elif ch == 'PP':
	lat_ch = 10.5; lon_ch = -88

"Viento"
pos_lon = np.where(lon == lon_ch)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == lat_ch)[0][0]
v   = archivo['v'][3  : pos_2017_12_31+1 : 4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
u   = archivo['u'][3  : pos_2017_12_31+1 : 4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
archivo.close()

wnd = np.sqrt(v*v+u*u)

"Remueve Ciclo Anual"
def cic_anual(data, dates):

	anom = np.zeros(len(data))
	for i in range(1,13):
		pos = np.where(dates.month == i)[0]
		mean = np.mean(data[pos])
		anom[pos] = data[pos] - mean
	return anom

ANOM = cic_anual(wnd, DATES)
ANOM = ANOM - np.min(ANOM) + 0.0000001 # Como el mínimo es número negativo, si lo resto, en realidad estoy sumando un positivo. El mínimo valor de la nueva serie, será 0.0000001

"Se guardan datos"

if np.all(ANOM > 0.0) == True: # Para confirmar que los datos si son positivos
	print "Datos positivos"
	wf     = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/24_expo_2018/datos_'+ch+'_EneDic_Anomalies_925.csv', 'w')
	writer = csv.writer(wf)
	for i, row in enumerate(ANOM):
		writer.writerow([row, TIME[i], DATES[i]])
	wf.close()
else:
	print "Datos no tienen mínimo en en cero"
