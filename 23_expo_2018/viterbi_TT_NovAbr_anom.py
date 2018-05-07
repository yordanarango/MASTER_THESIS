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
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[:] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas

"Fecha hasta donde se va a hacer HMM"
DATES          = DATES[3::4]
TIME           =  time[3::4]

"Selección de fechas en Noviembre, Diciembre, Enero, Febrero, Marzo, Abril"
DT  = []

leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales
years        = np.array([x for x in set(DATES.year)])

for i in years[:-1]:

    pos = np.where(DATES == pd.Timestamp(str(i)+'-11-01 18:00:00'))[0][0]

    if np.any(normal_years == i+1) == True:
    	DT.append(DATES[pos:pos+181])

    else:
    	DT.append(DATES[pos:pos+182])

dates = pd.DatetimeIndex(np.concatenate(DT))


"Lectura de Estados"
rf     = open("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/23_expo_2018/States_TT_NovAbr_anom.csv", 'r')
reader = csv.reader(rf)
states = [row for row in reader][1:]
rf.close()

states2 = np.array([int(x[1]) for x in states])
states3 = np.array([int(x[2]) for x in states])
states4 = np.array([int(x[3]) for x in states])

"Posiciones donde hay March 1 y no es bisiesto. Es decir febrero sólo llega hasta 28"
pos_mar1 = np.where((dates.day==1) & (dates.month==3))[0]

pos_mar1_feb28 = [] # Posiciones de Marzo 1 donde antes hay un febrero 28
for i in pos_mar1:
    if dates[i-1].day == 28:
        pos_mar1_feb28.append(i)

pos_mar1_feb28 = np.array(pos_mar1_feb28)

"Introduciendo nuevos valores de cero al vector de estados"
states2_new = np.insert(states2, pos_mar1_feb28, np.zeros(len(pos_mar1_feb28)))
states3_new = np.insert(states3, pos_mar1_feb28, np.zeros(len(pos_mar1_feb28)))
states4_new = np.insert(states4, pos_mar1_feb28, np.zeros(len(pos_mar1_feb28)))

"Número de estados deseados"
Nc = 2

" Matriz de estados, donde cada fila es un año de estados"
if Nc == 2:
    state_matrix = np.reshape(states2_new, (len(states2_new)/182, 182))
if Nc == 3:
    state_matrix = np.reshape(states3_new, (len(states3_new)/182, 182))
if Nc == 4:
    state_matrix = np.reshape(states4_new, (len(states4_new)/182, 182))
if Nc == 5:
    state_matrix = np.reshape(states5_new, (len(states5_new)/182, 182))


if Nc == 4:
    state_matrix[state_matrix == 2] = 33
    state_matrix[state_matrix == 3] = 22

    state_matrix[state_matrix == 22] = 2
    state_matrix[state_matrix == 33] = 3


"Ploteando con pcolor matriz de Viterbi de Estados"
# Dos estados
if Nc == 2:
    Colors  = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.)]
    bounds  = np.array([-0.5, 0.5, 1.5, 2.5])
    labelsx = ['ND', '1', '2']

# Tres estados
if Nc == 3:
    Colors = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0)]
    bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5])
    labelsx = ['ND', '1', '2', '3']

# Cuatro estados
if Nc == 4:
    Colors = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.)]
    bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    labelsx = ['ND', '1', '2', '3', '4']

# Cinco estados
if Nc == 5:
    Colors = [(255/255., 255/255., 255/255.), (255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.),
    (11/255., 83/255., 69/255.)]
    bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    labelsx = ['ND', '1', '2', '3', '4', '5']

cmap_name = 'my_list'
cm        = colors.LinearSegmentedColormap.from_list(cmap_name, Colors, N=Nc+1)

fig = plt.figure(figsize=(18,10))
ax  = fig.add_axes([0.06, 0.05, 0.93, 0.8])

norm = colors.BoundaryNorm(boundaries=bounds, ncolors=Nc+1)
pc  = ax.pcolor(state_matrix, norm=norm, cmap=cm)
cb  = fig.colorbar(pc, orientation='horizontal', ticks=np.arange(Nc+1), shrink=0.25, pad=0.08)
cb.set_label('State', size=17)
cb.ax.tick_params (labelsize=15)
cb.ax.set_position([0.15, -0.12, 0.2, 0.3])
cb.ax.set_xticklabels(labelsx)

for col in range(1, state_matrix.shape[1]+1):
	plt.axvline(x=col, ls='-', color='k', lw=1, alpha=1)

for row in range(1, state_matrix.shape[0]+1):
	plt.axhline(y=row, ls='-', color='k', lw=1, alpha=1)

x_ticks = np.arange(0, state_matrix.shape[1]+1)
y_ticks = np.arange(0, state_matrix.shape[0]+1)

my_y_ticks = ['79-80', '', '', '', '83-84', '', '', '', '87-88', '', '', '', '91-92', '', '', '',
              '95-96', '', '', '', '99-00', '', '', '', '03-04', '', '', '', '07-08', '', '', '',
              '11-12', '', '', '', '15-16']

my_x_ticks = ['Nov-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Dec-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Jan-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Feb-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Mar-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
              'Apr-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

plt.xticks(x_ticks, my_x_ticks, size=14)
plt.ylabel('Year', size=17)
plt.yticks(y_ticks, my_y_ticks, size=14)
plt.xlabel('Day', size=17)
ax.set_title(str(Nc)+' States - TT Wind Anomalies (Nov-Abr)', fontsize=18)

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/23_expo_2018/Vit_matrix_TT_NovAbr_'+str(Nc)+'st_anom.png', bbox_inches='tight', dpi=300)
