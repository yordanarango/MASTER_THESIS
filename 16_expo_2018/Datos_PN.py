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

archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[3::4] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas

"Viento"
pos_lon = np.where(lon == -79.5)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == 7)[0][0]
v   = archivo['v10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas
u   = archivo['u10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas
wnd = np.sqrt(v*v+u*u)

"Se seleccionan años NO bisiestos"
DT  = []
WND = []
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales. NO bisiestos

for i, d in enumerate(DATES):
	if np.any(normal_years == d.year):
		DT.append(d)
		WND.append(wnd[i])

dates = pd.DatetimeIndex(DT)
wind  = np.array(WND)

wf     = open("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/datos_PN_EneDic.csv", 'w')
writer = csv.writer(wf)
for row in wind:
	writer.writerow([row])
wf.close()



"################"
wf     = open("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/datos_PN_distribuciones.csv", 'w')
writer = csv.writer(wf)
for row in wnd:
	writer.writerow([row])
wf.close()
