# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 04:33:14 2018

@author: yordan
"""

import numpy as np
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.colors as colors
import datetime as dt
from dateutil.relativedelta import relativedelta
from pandas import Timestamp
import csv

##############################################################################################################
"Funciones"
class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def indices(Serie): # Serie que contiene dato cuando hay evento de chorro, y ceros cuando no se considera que hay evento de chorro
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

def ciclo_diurno_anual(matriz, fechas, len_lat, len_lon):
	#matriz : matriz de numpy de 3 dimensiones donde cada capa corresponde a una fecha en el vector de pandas "fechas"
	#fechas : objeto de pandas con las fechas que corresponden a cada una de las capas en matríz
	#len_lat: integer cantidad de pixeles en direccion meridional
	#len_lon: integer cantidad de pixeles en direccion zonal

	Dict_ciclo = {}
	for i, mes in enumerate(['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']):
		for j, hora in enumerate(['0', '6', '12', '18']):
			pos    = np.where((fechas.month == i+1 ) & (fechas.hour == int(hora)))[0]
			M      = np.zeros((len(pos), len_lat, len_lon))

			for k, l in enumerate(pos):
				M[k] = matriz[l]

			media = np.mean(M, axis=0)

			Dict_ciclo.update({mes+'_'+hora:media})

	return Dict_ciclo

"##############################################################################################################"

HMM = 3
# ch = u'Tehuantepec'; CH = u'TT'
# ch = u'Papagayo';    CH = u'PP'
ch = u'Panama';      CH = u'PN'

#
'DATOS RASI'
#
RASI_ch = []

for i in range(1998, 2012):
    mean_ch = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_'+ch+'_'+str(i)+'.nc')['WindSpeedMean'][:]
    RASI_ch.extend(mean_ch)

EVN_ch     = indices(RASI_ch)


#
"Fechas de cada evento"
#
Dates_RASI   = pd.date_range('1998-01-01', freq='6H', periods=len(RASI_ch))
Dates_ch_ini = [str(Dates_RASI[x[0]]) for x in EVN_ch] # Fechas en las que empezaron los eventos
Dates_ch_fin = [str(Dates_RASI[x[1]]) for x in EVN_ch] # Fechas en las que terminaron los eventos
Dates_ch_max = [str(Dates_RASI[(x[0] + np.where(RASI_ch[x[0]:x[1]+1] == np.max(RASI_ch[x[0]:x[1]+1])))[0][0]]) for x in EVN_ch] #fecha en que se da el máximo de cada evento


#
"Calculando duración de cada evento"
#
lon_ch = np.zeros(len(EVN_ch))

for i, ind in enumerate(EVN_ch):
    lon_ch[i] = ind[1]-ind[0]+1


#
"Máxima magnitud de cada evento"
#
max_ch = np.zeros(len(EVN_ch))

for i, ind in enumerate(EVN_ch):
    max_ch[i] = np.max(RASI_ch[ind[0]:ind[1]+1])


#
"Magnitud media de cada evento"
#
mean_ch = np.zeros(len(EVN_ch))


for i, ind in enumerate(EVN_ch):
    mean_ch[i] = np.mean(RASI_ch[ind[0]:ind[1]+1])


"Lectura de Estados"
rf     = open(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/17_expo_2018/States_'+CH+'.csv', 'r')
reader = csv.reader(rf)
states = [row for row in reader][1:]
rf.close()

if HMM == 2:
    st_sr = np.array([int(x[1]) for x in states])
elif HMM == 3:
    st_sr = np.array([int(x[2]) for x in states])
elif HMM == 4:
    st_sr = np.array([int(x[3]) for x in states])

if CH == 'TT' and HMM == 3:
    st_sr[st_sr == 2] = 33
    st_sr[st_sr == 3] = 22

    st_sr[st_sr == 22] = 2
    st_sr[st_sr == 33] = 3
elif CH == 'TT' and HMM == 4:
    st_sr[st_sr == 2] = 44
    st_sr[st_sr == 4] = 22

    st_sr[st_sr == 22] = 2
    st_sr[st_sr == 44] = 4
elif CH == 'PP' and HMM == 3:
    st_sr[st_sr == 3] = 22
    st_sr[st_sr == 2] = 33

    st_sr[st_sr == 33] = 3
    st_sr[st_sr == 22] = 2
elif CH == 'PP' and HMM == 4:
    st_sr[st_sr == 3] = 22
    st_sr[st_sr == 2] = 33

    st_sr[st_sr == 33] = 3
    st_sr[st_sr == 22] = 2
elif CH == 'PN' and HMM == 3:
    st_sr[st_sr == 2] = 33
    st_sr[st_sr == 3] = 22

    st_sr[st_sr == 33] = 3
    st_sr[st_sr == 22] = 2
elif CH == 'PN' and HMM == 4:
    st_sr[st_sr == 3] = 22
    st_sr[st_sr == 2] = 33

    st_sr[st_sr == 33] = 3
    st_sr[st_sr == 22] = 2

"#############################################       DATOS PARA FECHAS         ################################################"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)

"Fecha hasta donde se va a hacer HMM"
pos_2015_12_31 = np.where(DATES == Timestamp('2015-12-31 18:00:00'))[0][0]
DATES          = DATES[3 : pos_2015_12_31+1 : 4]

"##############################################################################################################################"

num_bins = 30
path     = u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/18_expo_2018'

for k in range(1, HMM+1):
    k
    pos   = np.where(st_sr == k)[0]
    dates = DATES[pos]

    dt_cumple = []
    ps_cumple = []
    for d, D in enumerate(dates):
        for f in range(len(Dates_ch_ini)):

            if Timestamp(Dates_ch_ini[f]) <= D and D <= Timestamp(Dates_ch_fin[f]):
                dt_cumple.append(D) # D = Fecha en la que había un estado k y además había un evento de chorro.
                ps_cumple.append(f) # f = Posición en eventos de RASI en que esto se cumple.
                break

    DUR = lon_ch [ps_cumple]/4.; print 'Duration', np.max(DUR), np.min(DUR) # Son datos cada 6 horas. De esta forma quedan en unidades de días
    MAG = mean_ch[ps_cumple];    print 'Magnitud', np.max(MAG), np.min(MAG)
    MAX = max_ch [ps_cumple];    print 'Maximo',   np.max(MAX), np.min(MAX)

    "HISTOGRAMA DE DURACION"
    n, bins, patches = plt.hist(DUR, num_bins, facecolor='green', alpha=0.5, rwidth=0.95)

    plt.xlabel(u'Duration [days]')
    plt.ylabel('Frequency')
    plt.xlim(0, 14)
    plt.title(CH+' Duration - '+str(k)+' (HMM'+str(HMM)+')')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(path+u'/Hist_Dur_'+CH+'_st'+str(k)+'HMM'+str(HMM)+'.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    "HISTOGRAMA DE MAGNITUD"
    n, bins, patches = plt.hist(MAG, num_bins, facecolor='red', alpha=0.5, rwidth=0.95)

    plt.xlabel(u'Speed [m/s]')
    plt.ylabel('Frequency')
    plt.xlim(6.5, 12)
    plt.title(CH+' Speed - '+str(k)+' (HMM'+str(HMM)+')')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(path+u'/Hist_Mag'+CH+'_st'+str(k)+'HMM'+str(HMM)+'.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    "HISTOGRAMA DE MAXIMO"
    n, bins, patches = plt.hist(MAX, num_bins, facecolor='blue', alpha=0.5, rwidth=0.95)

    plt.xlabel(u'Speed [m/s]')
    plt.ylabel('Frequency')
    plt.xlim(6.5, 14.5)
    plt.title(CH+' Speed-Max - '+str(k)+' (HMM'+str(HMM)+')')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig(path+u'/Hist_Max'+CH+'_st'+str(k)+'HMM'+str(HMM)+'.png', bbox_inches='tight', dpi=300)
    plt.close('all')
