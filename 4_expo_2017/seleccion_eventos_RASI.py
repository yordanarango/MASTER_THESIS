# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 04:33:14 2016

@author: yordan
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as pl
import numpy as np
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt
from scipy import linalg as la
import pandas as pd
import pickle
import xlrd
from scipy.stats.stats import pearsonr
import matplotlib.colors as colors
from scipy.stats.stats import pearsonr
from numpy import genfromtxt


#
"Funciones"
#
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

#
'DATOS RASI'
#

RASI_Tehuantepec = []
RASI_Papagayo    = []
RASI_Panama      = []

for i in range(1998, 2012):
	
	mean_TT = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Tehuantepec_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PP = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Papagayo_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PN = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Panama_'+str(i)+'.nc')['WindSpeedMean'][:]

	RASI_Tehuantepec.extend(mean_TT)
	RASI_Papagayo.extend(mean_PP)
	RASI_Panama.extend(mean_PN)

EVN_TT     = indices(RASI_Tehuantepec)
EVN_PP     = indices(RASI_Papagayo)
EVN_PN     = indices(RASI_Panama)

#
"Fechas de cada evento"
#
Dates_RASI   = pd.date_range('1998-01-01', freq='6H', periods=len(RASI_Tehuantepec))
Dates_TT_ini = [str(Dates_RASI[x[0]]) for x in EVN_TT]
Dates_PP_ini = [str(Dates_RASI[x[0]]) for x in EVN_PP]
Dates_PN_ini = [str(Dates_RASI[x[0]]) for x in EVN_PN]

Dates_TT_fin = [str(Dates_RASI[x[1]]) for x in EVN_TT]
Dates_PP_fin = [str(Dates_RASI[x[1]]) for x in EVN_PP]
Dates_PN_fin = [str(Dates_RASI[x[1]]) for x in EVN_PN]
#
"Calculando duración de cada evento"
#
lon_TT = np.zeros(len(EVN_TT))
lon_PP = np.zeros(len(EVN_PP))
lon_PN = np.zeros(len(EVN_PN))

for i, ind in enumerate(EVN_TT):
	lon_TT[i] = ind[1]-ind[0]+1

for i, ind in enumerate(EVN_PP):
	lon_PP[i] = ind[1]-ind[0]+1

for i, ind in enumerate(EVN_PN):
	lon_PN[i] = ind[1]-ind[0]+1
#
"Máxima magnitud de cada evento"
#
max_TT = np.zeros(len(EVN_TT))
max_PP = np.zeros(len(EVN_PP))
max_PN = np.zeros(len(EVN_PN))

for i, ind in enumerate(EVN_TT):
	max_TT[i] = np.max(RASI_Tehuantepec[ind[0]:ind[1]+1])

for i, ind in enumerate(EVN_PP):
	max_PP[i] = np.max(RASI_Papagayo[ind[0]:ind[1]+1])

for i, ind in enumerate(EVN_PN):
	max_PN[i] = np.max(RASI_Panama[ind[0]:ind[1]+1])

#
"Magnitud media de cada evento"
#
mean_TT = np.zeros(len(EVN_TT))
mean_PP = np.zeros(len(EVN_PP))
mean_PN = np.zeros(len(EVN_PN))

for i, ind in enumerate(EVN_TT):
	mean_TT[i] = np.mean(RASI_Tehuantepec[ind[0]:ind[1]+1])

for i, ind in enumerate(EVN_PP):
	mean_PP[i] = np.mean(RASI_Papagayo[ind[0]:ind[1]+1])

for i, ind in enumerate(EVN_PN):
	mean_PN[i] = np.mean(RASI_Panama[ind[0]:ind[1]+1])

#
"Dataframes con caracteristicas de los eventos"
#

caract_TT                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin'])
caract_TT['Duracion']      = lon_TT
caract_TT['Max']           = max_TT
caract_TT['Mean']          = mean_TT
caract_TT['Fechas inicio'] = Dates_TT_ini
caract_TT['Fechas fin']    = Dates_TT_fin
sort_by_lon_TT = caract_TT.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_TT = caract_TT.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_TT = caract_TT.sort_values(by = ['Mean'], ascending=[False])
lon_75_TT = np.percentile(lon_TT, 75); lon_50_TT = np.percentile(lon_TT, 50); lon_25_TT = np.percentile(lon_TT, 25)
max_75_TT = np.percentile(max_TT, 75); max_50_TT = np.percentile(max_TT, 50); max_25_TT = np.percentile(max_TT, 25)
mean_75_TT = np.percentile(mean_TT, 75); mean_50_TT = np.percentile(mean_TT, 50); mean_25_TT = np.percentile(mean_TT, 25)


caract_PP                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin'])
caract_PP['Duracion']      = lon_PP
caract_PP['Max']           = max_PP
caract_PP['Mean']          = mean_PP
caract_PP['Fechas inicio'] = Dates_PP_ini
caract_PP['Fechas fin']    = Dates_PP_fin
sort_by_lon_PP = caract_PP.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_PP = caract_PP.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_PP = caract_PP.sort_values(by = ['Mean'], ascending=[False])
lon_75_PP = np.percentile(lon_PP, 75); lon_50_PP = np.percentile(lon_PP, 50); lon_25_PP = np.percentile(lon_PP, 25)
max_75_PP = np.percentile(max_PP, 75); max_50_PP = np.percentile(max_PP, 50); max_25_PP = np.percentile(max_PP, 25)
mean_75_PP = np.percentile(mean_PP, 75); mean_50_PP = np.percentile(mean_PP, 50); mean_25_PP = np.percentile(mean_PP, 25)


caract_PN                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin'])
caract_PN['Duracion']      = lon_PN
caract_PN['Max']           = max_PN
caract_PN['Mean']          = mean_PN
caract_PN['Fechas inicio'] = Dates_PN_ini
caract_PN['Fechas fin']    = Dates_PN_fin
sort_by_lon_PN = caract_PP.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_PN = caract_PP.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_PN = caract_PP.sort_values(by = ['Mean'], ascending=[False])
lon_75_PN = np.percentile(lon_PN, 75); lon_50_PN = np.percentile(lon_PN, 50); lon_25_PN = np.percentile(lon_PN, 25)
max_75_PN = np.percentile(max_PN, 75); max_50_PN = np.percentile(max_PN, 50); max_25_PN = np.percentile(max_PN, 25)
mean_75_PN = np.percentile(mean_PN, 75); mean_50_PN = np.percentile(mean_PN, 50); mean_25_PN = np.percentile(mean_PN, 25)


#
"Fechas max"
#

Dmax_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dmax_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:10]].values


Dmax_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dmax_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:10]].values


Dmax_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dmax_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmax_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmax_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:10]].values



#
"Fechas mean"
#

Dmean_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dmean_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:10]].values


Dmean_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dmean_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:10]].values


Dmean_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dmean_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmean_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:10]].values
Dmean_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:10]].values


#
"Fechas Duracion"
#

Dlon_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dlon_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:10]].values


Dlon_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dlon_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:10]].values


Dlon_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dlon_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dlon_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:10]].values

# import glob
# import os
# import shutil
# for i in Dlon_25_50_TT_ini:
# 	os.mkdir('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/4_expo_2017/MAPAS_VIENTO/TT/DURACION/25_50/'+i, 0777)



MAX_INI = {'MAX_75_100_TT_INI':Dmax_75_100_TT_ini, 'MAX_50_75_TT_INI':Dmax_50_75_TT_ini, 'MAX_25_50_TT_INI':Dmax_25_50_TT_ini, 'MAX_75_100_PP_INI':Dmax_75_100_PP_ini, 'MAX_50_75_PP_INI':Dmax_50_75_PP_ini, 'MAX_25_50_PP_INI':Dmax_25_50_PP_ini, 'MAX_75_100_PN_INI':Dmax_75_100_PN_ini, 'MAX_50_75_PN_INI':Dmax_50_75_PN_ini, 'MAX_25_50_PN_INI':Dmax_25_50_PN_ini}
MAX_FIN = {'MAX_75_100_TT_FIN':Dmax_75_100_TT_fin, 'MAX_50_75_TT_FIN':Dmax_50_75_TT_fin, 'MAX_25_50_TT_FIN':Dmax_25_50_TT_fin, 'MAX_75_100_PP_FIN':Dmax_75_100_PP_fin, 'MAX_50_75_PP_FIN':Dmax_50_75_PP_fin, 'MAX_25_50_PP_FIN':Dmax_25_50_PP_fin, 'MAX_75_100_PN_FIN':Dmax_75_100_PN_fin, 'MAX_50_75_PN_FIN':Dmax_50_75_PN_fin, 'MAX_25_50_PN_FIN':Dmax_25_50_PN_fin}

DUR_INI = {'DUR_75_100_TT_INI':Dlon_75_100_TT_ini, 'DUR_50_75_TT_INI':Dlon_50_75_TT_ini, 'DUR_25_50_TT_INI':Dlon_25_50_TT_ini, 'DUR_75_100_PP_INI':Dlon_75_100_PP_ini, 'DUR_50_75_PP_INI':Dlon_50_75_PP_ini, 'DUR_25_50_PP_INI':Dlon_25_50_PP_ini, 'DUR_75_100_PN_INI':Dlon_75_100_PN_ini, 'DUR_50_75_PN_INI':Dlon_50_75_PN_ini, 'DUR_25_50_PN_INI':Dlon_25_50_PN_ini}
DUR_FIN = {'DUR_75_100_TT_FIN':Dlon_75_100_TT_fin, 'DUR_50_75_TT_FIN':Dlon_50_75_TT_fin, 'DUR_25_50_TT_FIN':Dlon_25_50_TT_fin, 'DUR_75_100_PP_FIN':Dlon_75_100_PP_fin, 'DUR_50_75_PP_FIN':Dlon_50_75_PP_fin, 'DUR_25_50_PP_FIN':Dlon_25_50_PP_fin, 'DUR_75_100_PN_FIN':Dlon_75_100_PN_fin, 'DUR_50_75_PN_FIN':Dlon_50_75_PN_fin, 'DUR_25_50_PN_FIN':Dlon_25_50_PN_fin}



