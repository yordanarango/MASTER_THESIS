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
ch = 'PP'

"Leyendo datos"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[:] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas

"Fecha hasta donde se va a hacer HMM"
pos_2015_12_31 = np.where(DATES == Timestamp('2015-12-31 18:00:00'))[0][0]
DATES          = DATES[3 : pos_2015_12_31+1 : 4]
TIME           =  time[3 : pos_2015_12_31+1 : 4]

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
v   = archivo['v10'][3  : pos_2015_12_31+1 : 4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
u   = archivo['u10'][3  : pos_2015_12_31+1 : 4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
wnd = np.sqrt(v*v+u*u)

"Se guardan datos"
wf     = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/17_expo_2018/datos_'+ch+'_EneDic.csv', 'w')
writer = csv.writer(wf)
for i, row in enumerate(wnd):
	writer.writerow([row, TIME[i], DATES[i]])
wf.close()
