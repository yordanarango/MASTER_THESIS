# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:34:25 2018

@author: yordan
"""
import numpy as np
import netCDF4 as nc
from netcdftime import utime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import csv
from pandas import Timestamp

"Leyendo datos"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area_925hPa.nc')

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

states2 = np.array([int(x[1]) for x in states])
states3 = np.array([int(x[2]) for x in states])
states4 = np.array([int(x[3]) for x in states])
states5 = np.array([int(x[4]) for x in states])

"Posiciones donde hay March 1 y no es bisiesto. Es decir febrero sólo llega hasta 28"
pos_mar1 = np.where((dates.day==1) & (dates.month==3) & (dates.hour==0))[0]

pos_mar1_feb28 = [] # Posiciones de Marzo 1 donde antes hay un febrero 28
for i in pos_mar1:
	if dates[i-1].day == 28:
		pos_mar1_feb28.append(i)

pos_mar1_feb28 = np.array(pos_mar1_feb28)

"Introduciendo nuevos valores de cero al vector de estados"
states2_new = states2
states3_new = states3
states4_new = states4
states5_new = states5
for i in range(4):

	states2_new = np.insert(states2_new, pos_mar1_feb28 + np.arange(1, len(pos_mar1_feb28) + 1) * i, 0)
	states3_new = np.insert(states3_new, pos_mar1_feb28 + np.arange(1, len(pos_mar1_feb28) + 1) * i, 0)
	states4_new = np.insert(states4_new, pos_mar1_feb28 + np.arange(1, len(pos_mar1_feb28) + 1) * i, 0)
	states5_new = np.insert(states5_new, pos_mar1_feb28 + np.arange(1, len(pos_mar1_feb28) + 1) * i, 0)


"Número de estados deseados"
Nc = 5

" Matriz de estados, donde cada fila es un año de estados"
if Nc == 2:
	state_matrix = np.reshape(states2_new, (len(states2_new)/(182*4), 182*4))
if Nc == 3:
	state_matrix = np.reshape(states3_new, (len(states3_new)/(182*4), 182*4))
if Nc == 4:
	state_matrix = np.reshape(states4_new, (len(states4_new)/(182*4), 182*4))
if Nc == 5:
	state_matrix = np.reshape(states5_new, (len(states5_new)/(182*4), 182*4))


if Nc == 4:
	state_matrix[state_matrix == 4] = 33
	state_matrix[state_matrix == 3] = 44

	state_matrix[state_matrix == 33] = 3
	state_matrix[state_matrix == 44] = 4

if Nc == 5:
	state_matrix[state_matrix == 5] = 33
	state_matrix[state_matrix == 3] = 55

	state_matrix[state_matrix == 33] = 3
	state_matrix[state_matrix == 55] = 5

state_matrix_nov = state_matrix[:,      : 30*4]
state_matrix_dec = state_matrix[:, 30*4 : 61*4]
state_matrix_jan = state_matrix[:, 61*4 : 92*4]
state_matrix_feb = state_matrix[:, 92*4 : 121*4]
state_matrix_mar = state_matrix[:, 121*4: 152*4]
state_matrix_apr = state_matrix[:, 152*4: 182*4]


"Ploteando con pcolor matriz de Viterbi de Estados"
# Dos estados
if Nc == 2:
	Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.)]
	bounds  = np.array([-0.5, 0.5, 1.5, 2.5])
	labelsx = ['ND', '1', '2']

# Tres estados
if Nc == 3:
	Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0)]
	bounds  = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
	labelsx = ['ND', '1', '2', '3']

# Cuatro estados
if Nc == 4:
	Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.)]
	bounds  = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
	labelsx = ['ND', '1', '2', '3', '4']

# Cinco estados
if Nc == 5:
	Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.),
	(11/255., 83/255., 69/255.)]
	bounds  = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
	labelsx = ['ND', '1', '2', '3', '4', '5']

# Seis estados
if Nc == 6:
	Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.),
	(11/255., 83/255., 69/255.), (185/255., 97/255., 18/255.)]
	bounds  = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
	labelsx = ['ND', '1', '2', '3', '4', '5', '6']


cmap_name = 'my_list'
cm        = colors.LinearSegmentedColormap.from_list(cmap_name, Colors, N=Nc+1)


for l, m in enumerate(['November', 'December', 'January', 'February', 'March', 'April']):

	if m == 'November':
		matrix     = state_matrix_nov

		my_x_ticks = ['Nov-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Nov-16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

	elif m == 'December':
		matrix     = state_matrix_dec

		my_x_ticks = ['Dec-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Dec-16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

	elif m == 'January':
		matrix     = state_matrix_jan

		my_x_ticks = ['Jan-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Jan-16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']


	elif m == 'February':
		matrix = state_matrix_feb

		my_x_ticks = ['Feb-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Feb-15', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

	elif m == 'March':
		matrix = state_matrix_mar

		my_x_ticks = ['Mar-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Mar-16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

	elif m == 'April':
		matrix = state_matrix_apr

		my_x_ticks = ['Apr-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					  'Apr-16', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
					     ''   , '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']


	fig = plt.figure(figsize=(18,10))
	ax  = fig.add_axes([0.06, 0.05, 0.93, 0.8])

	norm = colors.BoundaryNorm(boundaries=bounds, ncolors=Nc+1)
	pc   = ax.pcolor(matrix, norm=norm, cmap=cm)
	cb   = fig.colorbar(pc, orientation='horizontal', ticks=np.arange(Nc+1), shrink=0.25, pad=0.08)
	cb.set_label('State', size=17)
	cb.ax.tick_params (labelsize=15)
	cb.ax.set_position([0.15, -0.12, 0.2, 0.3])
	cb.ax.set_xticklabels(labelsx)

	for col in range(1, matrix.shape[1]+1):
		plt.axvline(x=col, ls='-', color='k', lw=1, alpha=1)

	for row in range(1, matrix.shape[0]+1):
		plt.axhline(y=row, ls='-', color='k', lw=1, alpha=1)

	x_ticks = np.arange(0, matrix.shape[1]+1)
	y_ticks = np.arange(0, matrix.shape[0]+1)

	my_y_ticks = ['79-80', '', '', '', '83-84', '', '', '', '87-88', '', '', '', '91-92', '', '', '',
				  '95-96', '', '', '', '99-00', '', '', '', '03-04', '', '', '', '07-08', '', '', '',
				  '11-12', '', '', '', '15-16', '']

	plt.xticks(x_ticks, my_x_ticks, size=14)
	plt.ylabel('Year', size=17)
	plt.yticks(y_ticks, my_y_ticks, size=14)
	plt.xlabel('Day', size=17)
	ax.set_title(str(Nc)+' States - TT Wind Anomalies (Nov-Abr)', fontsize=18)

	plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/viterbi/'+ str(l) +'_Vit_matrix_TT_NovAbr_'+str(Nc)+'st_anom_925_'+m[:3]+'.png', bbox_inches='tight', dpi=300)
