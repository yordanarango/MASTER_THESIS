# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:34:25 2018

@author: yordan
"""
import numpy as np
import netCDF4 as nc
from netcdftime import utime
from pandas import Timestamp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import csv
#from mpl_toolkits.basemap import Basemap

"FUNCIONES"

"###################################################################################################################################"

"Escoger chorro: TT, PP, PN"
ch = 'PN'

"Leyendo datos"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-360

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas) # Se toma una sóla hora del día de la velocidad

"Fecha hasta donde se va a hacer HMM"
DATES          = DATES[3::4]
TIME           =  time[3::4]

"coordenadas"
if ch == 'TT':
	lat_ch = 15  ; lon_ch = -95
elif ch == 'PP':
	lat_ch = 10.5; lon_ch = -88
elif ch == 'PN':
	lat_ch = 7   ; lon_ch = -79.5

"Viento"
pos_lon = np.where(lon == lon_ch)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == lat_ch)[0][0]
v   = archivo['v10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
u   = archivo['u10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas
wnd = np.sqrt(v*v+u*u)

"Selección de datos en Noviembre, Diciembre, Enero, Febrero, Marzo"
DT  = []
WND = []
TM  = []

leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales
years        = np.array([x for x in set(DATES.year)])

for i in years[:-1]:

    pos = np.where(DATES == pd.Timestamp(str(i)+'-11-01 18:00:00'))[0][0]

    if np.any(normal_years == i+1) == True:
    	DT.append(DATES[pos:pos+151])
    	WND.append(wnd[pos:pos+151])
        TM.append(TIME[pos:pos+151])
    else:
    	DT.append(DATES[pos:pos+152])
    	WND.append(wnd[pos:pos+152])
        TM.append(TIME[pos:pos+152])

dates = pd.DatetimeIndex(np.concatenate(DT))
wind  = np.concatenate(WND)
tm    = np.concatenate(TM)

"Se guardan datos"
wf     = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/20_expo_2018/datos_'+ch+'_NovMar.csv', 'w')
writer = csv.writer(wf)
for i, row in enumerate(wind):
    writer.writerow([row, tm[i], dates[i]])
wf.close()
