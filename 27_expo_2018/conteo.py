# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:34:25 2018

@author: yordan
"""
import numpy as np
import netCDF4 as nc
from netcdftime import utime
import pandas as pd
import csv
from pandas import Timestamp
import math
from windrose import WindroseAxes
import matplotlib.pyplot as plt



"##################################     FUNCIONES    ####################################"
def indices(Serie): # Devuelve posiciones donde empieza y termina un evento
	# Serie que contiene dato cuando hay evento de chorro, y ceros cuando no se considera que hay evento de chorro

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


def histograma(data_u, data_v, color, unds, title, path, xlim_min, xlim_max, ylim_max, num_bins=30):

	# data_u: Lista con datos de u con los que se va a calcular velocidad del viento
	# data_v: Lista con datos de v con los que se va a calcular velocidad del viento
	# color:  string del color que va a llevar el histograma
	# unds:   String de las unidades de la variable que se está manejando
	# title:  String del titulo del histograma
	# path:   String de la ruta con el nombre del archivo
	# xlim_min: Float del límite inferior del histograma
	# xlim_min: Float del límite superior del histograma
	# num_bins: Entero del número de bins que va a tener el histograma

	u   = np.array(data_u)
	v   = np.array(data_v)
	spd = np.sqrt(u*u + v*v)

	weights  = np.ones_like(spd)/float(len(spd))

	delta = (xlim_max - xlim_min)/float(num_bins)
	BINS  = xlim_min + np.arange(num_bins+1) * delta

	plt.hist(spd, bins=BINS, weights=weights, facecolor=color, rwidth=0.95)
	plt.xlabel(unds)
	plt.ylabel(u'Frequency')
	plt.xlim(xlim_min-1, xlim_max+1)
	plt.ylim(0, ylim_max)
	plt.title(title)
	plt.savefig(path + '.png', bbox_inches='tight', dpi=300)
	plt.close('all')


def direction(U, V):
	"U: x wind component serie of numpy"
	"V: y wind component serie of numpy"

	#return
	"d: wind direction serie of numpy"

	d = []

	for u, v in zip(U, V):

		if u > 0 and v > 0:
			#print (np.arctan(u/v)*180/math.pi)
			D = (np.arctan(u/v)*180/math.pi)
			d.append(D)
		
		elif u > 0 and v < 0:
			#print (np.arctan(u/v)*180/math.pi) + 360
			D = (np.arctan(u/v)*180/math.pi) + 360
			d.append(D)

		elif u < 0 and v > 0:
			#print (np.arctan(u/v)*180/math.pi) + 180
			D = (np.arctan(u/v)*180/math.pi) + 180
			d.append(D)

		elif u < 0 and v < 0:
			#print (np.arctan(u/v)*180/math.pi) + 180
			D = (np.arctan(u/v)*180/math.pi) + 180
			d.append(D)

		elif u > 0 and v == 0:
			D = 0.
			d.append(D)

		elif u < 0 and v == 0:
			D = 180.
			d.append(D)

		elif u == 0 and v > 0:
			D = 90.
			d.append(D)

		elif u == 0 and v < 0:
			D = 270.
			d.append(D)

	return np.array(d)


def rosa(data_u, data_v, path, titulo):

	# data_u: Lista con datos de u con los que se va a calcular velocidad del viento
	# data_v: Lista con datos de v con los que se va a calcular velocidad del viento

	# Devuelce rosa de vientos del viento caracterizado por data_u y data_v

	u   = np.array(data_u)
	v   = np.array(data_v)
	spd = np.sqrt(u*u + v*v)
	
	Dir   = direction(u, v)
	 
	ax = WindroseAxes.from_ax()
	ax.bar(Dir, spd, normed=True, opening=0.8, edgecolor='white')
	ax.set_legend()
	ax.set_title(titulo, fontsize=15)
	plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
	plt.close('all')

"#########################################################################################"


"DATOS RASI"

RASI_Tehuantepec = []
RASI_Papagayo    = []
RASI_Panama      = []

for i in range(1998, 2012):

	mean_TT = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Tehuantepec_'+str(i)+'.nc')
	mean_PP = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Papagayo_'+str(i)+'.nc')
	mean_PN = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Panama_'+str(i)+'.nc')

	RASI_Tehuantepec.extend(mean_TT['WindSpeedMean'][:])
	RASI_Papagayo.extend(mean_PP['WindSpeedMean'][:])
	RASI_Panama.extend(mean_PN['WindSpeedMean'][:])

	mean_TT.close()
	mean_PP.close()
	mean_PN.close()

EVN_TT     = indices(RASI_Tehuantepec)
EVN_PP     = indices(RASI_Papagayo)
EVN_PN     = indices(RASI_Panama)


"FECHAS DE CADA EVENTO EN RASI"

Dates_RASI   = pd.date_range('1998-01-01', freq='6H', periods=len(RASI_Tehuantepec))

Dates_TT_ini = [str(Dates_RASI[x[0]]) for x in EVN_TT] # Fechas en las que empezaron los eventos
Dates_PP_ini = [str(Dates_RASI[x[0]]) for x in EVN_PP] # Fechas en las que empezaron los eventos
Dates_PN_ini = [str(Dates_RASI[x[0]]) for x in EVN_PN] # Fechas en las que empezaron los eventos

Dates_TT_fin = [str(Dates_RASI[x[1]]) for x in EVN_TT] # Fechas en las que terminaron los eventos
Dates_PP_fin = [str(Dates_RASI[x[1]]) for x in EVN_PP] # Fechas en las que terminaron los eventos
Dates_PN_fin = [str(Dates_RASI[x[1]]) for x in EVN_PN] # Fechas en las que terminaron los eventos


"DATOS HMM"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area_925hPa.nc')
lat     = archivo.variables['latitude'][:]
lon     = archivo.variables['longitude'][:]-360

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas) 

"Fecha hasta donde se va a hacer HMM"
pos_2017_04_30 = np.where(DATES == Timestamp('2017-04-30 18:00:00'))[0][0]
DATES          = DATES[: pos_2017_04_30+1] # Contrario a como se venía haciendo (cada 24 horas a las 6 de la tarde), se toman todos los datos
TIME           =  time[: pos_2017_04_30+1] # Contrario a como se venía haciendo (cada 24 horas a las 6 de la tarde), se toman todos los datos

"Selección de fechas en Noviembre, Diciembre, Enero, Febrero, Marzo, Abril"
DT  = []

leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales
years        = np.array([x for x in set(DATES.year)])

for i in years[:-1]:

	pos = np.where(DATES == pd.Timestamp(str(i)+'-11-01 00:00:00'))[0][0]

	if np.any(normal_years == i+1) == True:
		DT.append(DATES[pos:pos+181*4])

	else:
		DT.append(DATES[pos:pos+182*4])

dates = pd.DatetimeIndex(np.concatenate(DT))


"Lectura de Estados"
rf     = open("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/States_TT_NovAbr_anom_925.csv", 'r')
reader = csv.reader(rf)
states = [row for row in reader][1:]
rf.close()

Nc      = 5
states2 = np.array([int(x[1]) for x in states])
states3 = np.array([int(x[2]) for x in states])
states4 = np.array([int(x[3]) for x in states])
states5 = np.array([int(x[4]) for x in states])

if Nc == 5:

	state_serie = states5

	state_serie[state_serie == 5] = 33
	state_serie[state_serie == 3] = 55

	state_serie[state_serie == 33] = 3
	state_serie[state_serie == 55] = 5

"CONTEO"

matrix_count = np.zeros((4,5)) # Matriz donde las filas son las horas 00, 06, 12 y 18, y las columnas don los estados 1, 2, 3, 4, 5.

ser_0_1_u = []; ser_6_1_u = []; ser_12_1_u = []; ser_18_1_u = []
ser_0_2_u = []; ser_6_2_u = []; ser_12_2_u = []; ser_18_2_u = []
ser_0_3_u = []; ser_6_3_u = []; ser_12_3_u = []; ser_18_3_u = []
ser_0_4_u = []; ser_6_4_u = []; ser_12_4_u = []; ser_18_4_u = []
ser_0_5_u = []; ser_6_5_u = []; ser_12_5_u = []; ser_18_5_u = []

ser_0_1_v = []; ser_6_1_v = []; ser_12_1_v = []; ser_18_1_v = []
ser_0_2_v = []; ser_6_2_v = []; ser_12_2_v = []; ser_18_2_v = []
ser_0_3_v = []; ser_6_3_v = []; ser_12_3_v = []; ser_18_3_v = []
ser_0_4_v = []; ser_6_4_v = []; ser_12_4_v = []; ser_18_4_v = []
ser_0_5_v = []; ser_6_5_v = []; ser_12_5_v = []; ser_18_5_v = []


"coordenadas"
#if ch == 'TT':
lat_ch = 15  ; lon_ch = -95
# elif ch == 'PP':
# 	lat_ch = 10.5; lon_ch = -88
# elif ch == 'PN':
# 	lat_ch = 7   ; lon_ch = -79.5

pos_lat = np.where(lat == lat_ch)[0][0]
pos_lon = np.where(lon == lon_ch)[0][0] 


for ini, fin in zip(Dates_TT_ini, Dates_TT_fin): # Se recorren las fechas de cada uno de las eventos en RASI

	dts_RASI = pd.date_range(ini, fin, freq='6H') # Se forma un vector con las fechas que componen el evento 

	for d in dts_RASI: # Se recorren las fechas que componen en el evento

		if np.any(np.array((11,12,1,2,3,4)) == d.month) == True: # La fecha tiene que estar entre los meses de Noviembre a Abril
			
			pos_hmm = np.where(dates == d)[0][0] # Se busca la posición donde el vector de estados con la fecha del evento
			pos_ERA = np.where(DATES == d)[0][0] # Se busca la posición donde los datos de ERA cumplen con la fecha d 

			stt     = state_serie[pos_hmm]      # Se extrae el estado

			matrix_count[d.hour//6, stt-1] = matrix_count[d.hour//6, stt-1] + 1 # Se suma una unidad en la posición indicada

			if stt == 1:
				if d.hour == 0:
					ser_0_1_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 00 horas, estado 1
					ser_0_1_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 00 horas, estado 1
				elif d.hour == 6:
					ser_6_1_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 06 horas, estado 1
					ser_6_1_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 06 horas, estado 1
				elif d.hour == 12:
					ser_12_1_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 12 horas, estado 1
					ser_12_1_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 12 horas, estado 1
				elif d.hour == 18:
					ser_18_1_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 18 horas, estado 1
					ser_18_1_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 18 horas, estado 1
			
			if stt == 2:
				if d.hour == 0:
					ser_0_2_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 00 horas, estado 2
					ser_0_2_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 00 horas, estado 2
				elif d.hour == 6:
					ser_6_2_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 06 horas, estado 2
					ser_6_2_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 06 horas, estado 2
				elif d.hour == 12:
					ser_12_2_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 12 horas, estado 2
					ser_12_2_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 12 horas, estado 2
				elif d.hour == 18:
					ser_18_2_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 18 horas, estado 2
					ser_18_2_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 18 horas, estado 2

			if stt == 3:
				if d.hour == 0:
					ser_0_3_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 00 horas, estado 3
					ser_0_3_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 00 horas, estado 3
				elif d.hour == 6:
					ser_6_3_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 06 horas, estado 3
					ser_6_3_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 06 horas, estado 3
				elif d.hour == 12:
					ser_12_3_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 12 horas, estado 3
					ser_12_3_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 12 horas, estado 3
				elif d.hour == 18:
					ser_18_3_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 18 horas, estado 3
					ser_18_3_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 18 horas, estado 3

			if stt == 4:
				if d.hour == 0:
					ser_0_4_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 00 horas, estado 4
					ser_0_4_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 00 horas, estado 4
				elif d.hour == 6:
					ser_6_4_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 06 horas, estado 4
					ser_6_4_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 06 horas, estado 4
				elif d.hour == 12:
					ser_12_4_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 12 horas, estado 4
					ser_12_4_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 12 horas, estado 4
				elif d.hour == 18:
					ser_18_4_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 18 horas, estado 4
					ser_18_4_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 18 horas, estado 4

			if stt == 5:			
				if d.hour == 0:
					ser_0_5_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 00 horas, estado 5
					ser_0_5_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 00 horas, estado 5
				elif d.hour == 6:
					ser_6_5_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 06 horas, estado 5
					ser_6_5_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 06 horas, estado 5
				elif d.hour == 12:
					ser_12_5_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 12 horas, estado 5
					ser_12_5_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 12 horas, estado 5
				elif d.hour == 18:
					ser_18_5_u.append(archivo['u'][pos_ERA, pos_lat, pos_lon]) # u 18 horas, estado 5
					ser_18_5_v.append(archivo['v'][pos_ERA, pos_lat, pos_lon]) # v 18 horas, estado 5


"Histograma velocidad del viento"
# path  = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/hist_rosa/'
# title = 'Wind Speed Histogram - '

# histograma(ser_0_1_u,  ser_0_1_v, '#641e16',  'm/s', title+'00:00 hrs (state 1)', path+'hist_0_1',  1.0, 37.0, 0.15, 30)
# histograma(ser_6_1_u,  ser_6_1_v, '#922b21',  'm/s', title+'06:00 hrs (state 1)', path+'hist_6_1',  1.0, 37.0, 0.15, 30)
# histograma(ser_12_1_u, ser_12_1_v, '#c0392b', 'm/s', title+'12:00 hrs (state 1)', path+'hist_12_1', 1.0, 37.0, 0.15, 30)
# histograma(ser_18_1_u, ser_18_1_v, '#d98880', 'm/s', title+'18:00 hrs (state 1)', path+'hist_18_1', 1.0, 37.0, 0.15, 30)

# histograma(ser_0_2_u,  ser_0_2_v, '#6e2c00',  'm/s', title+'00:00 hrs (state 2)', path+'hist_0_2',  1.0, 37.0, 0.2, 30)
# histograma(ser_6_2_u,  ser_6_2_v, '#a04000',  'm/s', title+'06:00 hrs (state 2)', path+'hist_6_2',  1.0, 37.0, 0.2, 30)
# histograma(ser_12_2_u, ser_12_2_v, '#d35400', 'm/s', title+'12:00 hrs (state 2)', path+'hist_12_2', 1.0, 37.0, 0.2, 30)
# histograma(ser_18_2_u, ser_18_2_v, '#e59866', 'm/s', title+'18:00 hrs (state 2)', path+'hist_18_2', 1.0, 37.0, 0.2, 30)

# #histograma(ser_0_3_u,  ser_0_3_v,  '#17202a', 'm/s', title+'00:00 hrs (state 3)', path+'hist_0_3',  1.0, 37.0, 0.3, 30)
# #histograma(ser_6_3_u,  ser_6_3_v,  '#212f3d', 'm/s', title+'06:00 hrs (state 3)', path+'hist_6_3',  1.0, 37.0, 0.3, 30)
# histograma(ser_12_3_u, ser_12_3_v, '#2c3e50', 'm/s', title+'12:00 hrs (state 3)', path+'hist_12_3', 1.0, 37.0, 0.3, 30)
# histograma(ser_18_3_u, ser_18_3_v, '#808b96', 'm/s', title+'18:00 hrs (state 3)', path+'hist_18_3', 1.0, 37.0, 0.3, 30)

# histograma(ser_0_4_u,  ser_0_4_v, '#145a32',  'm/s', title+'00:00 hrs (state 4)', path+'hist_0_4',  1.0, 37.0, 0.41, 30)
# histograma(ser_6_4_u,  ser_6_4_v, '#1e8449',  'm/s', title+'06:00 hrs (state 4)', path+'hist_6_4',  1.0, 37.0, 0.41, 30)
# histograma(ser_12_4_u, ser_12_4_v, '#27ae60', 'm/s', title+'12:00 hrs (state 4)', path+'hist_12_4', 1.0, 37.0, 0.41, 30)
# histograma(ser_18_4_u, ser_18_4_v, '#7dcea0', 'm/s', title+'18:00 hrs (state 4)', path+'hist_18_4', 1.0, 37.0, 0.41, 30)

# histograma(ser_0_5_u,  ser_0_5_v, '#1b4f72',  'm/s', title+'00:00 hrs (state 5)', path+'hist_0_5',  1.0, 37.0, 0.27, 30)
# histograma(ser_6_5_u,  ser_6_5_v, '#2874a6',  'm/s', title+'06:00 hrs (state 5)', path+'hist_6_5',  1.0, 37.0, 0.27, 30)
# histograma(ser_12_5_u, ser_12_5_v, '#3498db', 'm/s', title+'12:00 hrs (state 5)', path+'hist_12_5', 1.0, 37.0, 0.27, 30)
# histograma(ser_18_5_u, ser_18_5_v, '#85c1e9', 'm/s', title+'18:00 hrs (state 5)', path+'hist_18_5', 1.0, 37.0, 0.27, 30)



"Rosa de vientos"

path  = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/hist_rosa/'
#path  = '/home/yordan/Escritorio/'

rosa(ser_0_1_u,  ser_0_1_v,  path+'rosa_0_1', 'State 1 - Hour 00:00')
rosa(ser_6_1_u,  ser_6_1_v,  path+'rosa_6_1', 'State 1 - Hour 06:00')
rosa(ser_12_1_u, ser_12_1_v, path+'rosa_12_1', 'State 1 - Hour 12:00')
rosa(ser_18_1_u, ser_18_1_v, path+'rosa_18_1', 'State 1 - Hour 18:00')

rosa(ser_0_2_u,  ser_0_2_v,  path+'rosa_0_2', 'State 2 - Hour 00:00')
rosa(ser_6_2_u,  ser_6_2_v,  path+'rosa_6_2', 'State 2 - Hour 06:00')
rosa(ser_12_2_u, ser_12_2_v, path+'rosa_12_2', 'State 2 - Hour 12:00')
rosa(ser_18_2_u, ser_18_2_v, path+'rosa_18_2', 'State 2 - Hour 18:00')

#rosa(ser_0_3_u,  ser_0_3_v,  path+'rosa_0_3', 'State 3 - Hour 00:00')
#rosa(ser_6_3_u,  ser_6_3_v,  path+'rosa_6_3', 'State 3 - Hour 06:00')
rosa(ser_12_3_u, ser_12_3_v, path+'rosa_12_3', 'State 3 - Hour 12:00')
rosa(ser_18_3_u, ser_18_3_v, path+'rosa_18_3', 'State 3 - Hour 18:00')

rosa(ser_0_4_u,  ser_0_4_v,  path+'rosa_0_4', 'State 4 - Hour 00:00')
rosa(ser_6_4_u,  ser_6_4_v,  path+'rosa_6_4', 'State 4 - Hour 06:00')
rosa(ser_12_4_u, ser_12_4_v, path+'rosa_12_4', 'State 4 - Hour 12:00')
rosa(ser_18_4_u, ser_18_4_v, path+'rosa_18_4', 'State 4 - Hour 18:00')

rosa(ser_0_5_u,  ser_0_5_v,  path+'rosa_0_5', 'State 5 - Hour 00:00')
rosa(ser_6_5_u,  ser_6_5_v,  path+'rosa_6_5', 'State 5 - Hour 06:00')
rosa(ser_12_5_u, ser_12_5_v, path+'rosa_12_5', 'State 5 - Hour 12:00')
rosa(ser_18_5_u, ser_18_5_v, path+'rosa_18_5', 'State 5 - Hour 18:00')


