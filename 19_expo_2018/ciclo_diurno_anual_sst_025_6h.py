# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 13:34:25 2018

@author: yordan
"""
import numpy as np
import pickle
import pandas as pd
from netcdftime import utime
import netCDF4 as nc

def ciclo_diurno_anual(matriz, fechas, len_lat, len_lon):
	#matriz : matriz de numpy de 3 dimensiones donde cada capa corresponde a una fecha en el vector de pandas "fechas"
	#fechas : objeto de pandas con las fechas que corresponden a cada una de las capas en matríz
	#len_lat: integer cantidad de pixeles en direccion meridional
	#len_lon: integer cantidad de pixeles en direccion zonal

	#return: devuelve diccionario con ciclo diuno para cada mes
	Dict_ciclo = {}
	for i, mes in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']):
		for j, hora in enumerate(['0', '6', '12', '18']):
			pos    = np.where((fechas.month == i+1 ) & (fechas.hour == int(hora)))[0]
			M      = np.zeros((len(pos), len_lat, len_lon))

			for k, l in enumerate(pos):
				M[k] = matriz[l]

			media = np.mean(M, axis=0)

			Dict_ciclo.update({mes+'_'+hora:media})

	return Dict_ciclo

"Se leen datos de presión a resolución de 0.25 grados"
archivo = nc.Dataset('.../SST_EOFs.nc')
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-360

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[:]

"Presión"
sst   = archivo['sst'][:]

"Se calcula ciclo anual"
CICLO_SST = ciclo_diurno_anual(sst, DATES, len(lat), len(lon))


punto_bin = open('.../ciclo_diurno_anual_sst_025_6h.bin','wb')
pickle.dump(CICLO_SST, punto_bin)
punto_bin = open('.../ciclo_diurno_anual_sst_025_6h.bin','wb')
pickle.dump(CICLO_SST, punto_bin)
