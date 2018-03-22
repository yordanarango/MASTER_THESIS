# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 13:34:25 2018

@author: yordan
"""

import numpy as np
import netCDF4 as nc
from netcdftime import utime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import csv
"FUNCIONES"
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

"###################################################################################################################################"

archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
Variables = [x for x in archivo.variables]

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[3::4] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas

"Se seleccionan años NO bisiestos"
DT  = []
WND = []
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales. NO bisiestos

"Lectura de Estados"

rf     = open("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/States_PN.csv", 'r')
reader = csv.reader(rf)
states = [row for row in reader]

rf.close()

"Numero de estados deseados"
Nc     = 2 # Número de estados
hidden_states = [int(x[Nc-1]) for x in states[1:]] # el primero es un título

" Matriz de estados, donde cada fila es un año de estados"
state_matrix = np.reshape(hidden_states, (len(normal_years), len(hidden_states)/len(normal_years)))
# state_matrix[state_matrix == 3] = 22
# state_matrix[state_matrix == 2] = 33
#
# state_matrix[state_matrix == 22] = 2
# state_matrix[state_matrix == 33] = 3

"Ploteando con pcolor matriz de Viterbi de Estados"
# Dos estados
if Nc == 2:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.)]
	bounds = np.array([0.5, 1.5, 2.5])

# Tres estados
if Nc == 3:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5])

# Cuatro estados
if Nc == 4:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

# Cinco estados
if Nc == 5:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.),
			  (11/255., 83/255., 69/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

cmap_name = 'my_list'
cm        = colors.LinearSegmentedColormap.from_list(cmap_name, Colors, N=Nc)

fig = plt.figure(figsize=(18,10))
ax  = fig.add_axes([0.06, 0.05, 0.93, 0.8])

norm = colors.BoundaryNorm(boundaries=bounds, ncolors=Nc)
pc  = ax.pcolor(state_matrix, norm=norm, cmap=cm)
cb  = fig.colorbar(pc, orientation='horizontal', ticks=np.arange(Nc)+1, shrink=0.25, pad=0.08)
cb.set_label('State', size=17)
cb.ax.tick_params (labelsize=15)
cb.ax.set_position([0.15, -0.12, 0.2, 0.3])

for col in range(1, state_matrix.shape[1]+1):
	plt.axvline(x=col, ls='-', color='k', lw=1, alpha=1)
	#if col == 106: # 16 de Abril
		#plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 181: # 30 de Junio
		#plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 228: # 16 de Agosto
		#plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 304: # 31 de Octubre
		#plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)

for row in range(1, state_matrix.shape[0]+1):
	plt.axhline(y=row, ls='-', color='k', lw=1, alpha=1)

x_ticks = np.arange(0, state_matrix.shape[1]+1)
y_ticks = np.arange(0, state_matrix.shape[0]+1)

my_y_ticks = ['1979', '1981', '', '', '1895', '', '', '1989', '', '', '1993', '', '', '1997', '', '', '2001', '', '', '2005', '', '',
              '2009', '', '', '2013', '', '']

my_x_ticks = ['Jan-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Feb-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Mar-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Apr-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'May-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Jun-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Jul-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Aug-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Sep-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Oct-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Nov-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Dic-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

plt.xticks(x_ticks, my_x_ticks, size=14)
plt.ylabel('Year', size=17)
plt.yticks(y_ticks, my_y_ticks, size=14)
plt.xlabel('Day', size=17)
ax.set_title(str(Nc)+' States - Wind in PN', fontsize=18)

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/Viterbi_matrix_PN_EneDic_'+str(Nc)+'st.png', bbox_inches='tight', dpi=300)
