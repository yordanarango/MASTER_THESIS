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
DATES   = pd.DatetimeIndex(fechas)

"Fecha hasta donde se va a hacer HMM"
pos_2017_04_30 = np.where(DATES == Timestamp('2017-04-30 18:00:00'))[0][0]
DATES          = DATES[:pos_2017_04_30+1] # Contrario a como se venía haciendo (cada 24 horas a las 6 de la tarde), se toman todos los datos
TIME           = time[:pos_2017_04_30+1]  # Contrario a como se venía haciendo (cada 24 horas a las 6 de la tarde), se toman todos los datos

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
v       = archivo['v'][:pos_2017_04_30+1, pos_lat, pos_lon] 
u       = archivo['u'][:pos_2017_04_30+1, pos_lat, pos_lon] 
wnd     = np.sqrt(v*v+u*u)

"Selección de datos en Noviembre, Diciembre, Enero, Febrero, Marzo, Abril"
DT  = []
WND = []
TM  = []

leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales
years        = np.array([x for x in set(DATES.year)])

for i in years[:-1]:

    pos = np.where(DATES == pd.Timestamp(str(i)+'-11-01 00:00:00'))[0][0]

    if np.any(normal_years == i+1) == True:
    	DT.append(DATES[pos:pos+181*4])
    	WND.append(wnd[pos:pos+181*4])
        TM.append(TIME[pos:pos+181*4])
    else:
    	DT.append(DATES[pos:pos+182*4])
    	WND.append(wnd[pos:pos+182*4])
        TM.append(TIME[pos:pos+182*4])

dates = pd.DatetimeIndex(np.concatenate(DT))
wind  = np.concatenate(WND)
tm    = np.concatenate(TM)

"Remueve Ciclo Anual"
def cic_anual(data, dates):

	anom = np.zeros(len(data))
	for i in range(1,13):
		pos = np.where(dates.month == i)[0]
		mean = np.mean(data[pos])
		anom[pos] = data[pos] - mean
	return anom

ANOM = cic_anual(wind, dates)
ANOM = ANOM - np.min(ANOM) + 0.0000001 # Como el mínimo es número negativo, si lo resto, en realidad estoy sumando un positivo. El mínimo valor de la nueva serie, será 0.0000001

"Se guardan datos"
wf     = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/datos_'+ch+'_NovAbr_anom_925_6h.csv', 'w')
writer = csv.writer(wf)
for i, row in enumerate(ANOM):
    writer.writerow([row, tm[i], dates[i]])
wf.close()