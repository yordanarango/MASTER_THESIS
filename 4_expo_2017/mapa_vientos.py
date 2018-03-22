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
from dateutil.relativedelta import relativedelta



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


Dmax_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:5]].values
Dmax_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:5]].values
Dmax_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:5]].values
Dmax_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:5]].values
Dmax_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:5]].values
Dmax_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:5]].values


Dmax_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:5]].values
Dmax_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:5]].values
Dmax_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:5]].values
Dmax_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:5]].values
Dmax_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:5]].values
Dmax_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:5]].values



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

Dlon_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:5]].values
Dlon_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:5]].values
Dlon_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:10]].values
Dlon_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:10]].values


Dlon_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:5]].values
Dlon_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:5]].values
Dlon_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:5]].values
Dlon_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:5]].values


Dlon_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:5]].values
Dlon_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:5]].values
Dlon_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:5]].values
Dlon_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:5]].values
Dlon_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:5]].values

# import glob
# import os
# import shutil
# for i in Dlon_25_50_PN_ini:
# 	os.mkdir('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/4_expo_2017/MAPAS_VIENTO/PN/DURACION/25_50/'+i, 0777)



MAX_INI = {'MAX_75_100_TT_INI':Dmax_75_100_TT_ini, 'MAX_50_75_TT_INI':Dmax_50_75_TT_ini, 'MAX_25_50_TT_INI':Dmax_25_50_TT_ini, 'MAX_75_100_PP_INI':Dmax_75_100_PP_ini, 'MAX_50_75_PP_INI':Dmax_50_75_PP_ini, 'MAX_25_50_PP_INI':Dmax_25_50_PP_ini, 'MAX_75_100_PN_INI':Dmax_75_100_PN_ini, 'MAX_50_75_PN_INI':Dmax_50_75_PN_ini, 'MAX_25_50_PN_INI':Dmax_25_50_PN_ini}
MAX_FIN = {'MAX_75_100_TT_FIN':Dmax_75_100_TT_fin, 'MAX_50_75_TT_FIN':Dmax_50_75_TT_fin, 'MAX_25_50_TT_FIN':Dmax_25_50_TT_fin, 'MAX_75_100_PP_FIN':Dmax_75_100_PP_fin, 'MAX_50_75_PP_FIN':Dmax_50_75_PP_fin, 'MAX_25_50_PP_FIN':Dmax_25_50_PP_fin, 'MAX_75_100_PN_FIN':Dmax_75_100_PN_fin, 'MAX_50_75_PN_FIN':Dmax_50_75_PN_fin, 'MAX_25_50_PN_FIN':Dmax_25_50_PN_fin}

DUR_INI = {'DUR_75_100_TT_INI':Dlon_75_100_TT_ini, 'DUR_50_75_TT_INI':Dlon_50_75_TT_ini, 'DUR_25_50_TT_INI':Dlon_25_50_TT_ini, 'DUR_75_100_PP_INI':Dlon_75_100_PP_ini, 'DUR_50_75_PP_INI':Dlon_50_75_PP_ini, 'DUR_25_50_PP_INI':Dlon_25_50_PP_ini, 'DUR_75_100_PN_INI':Dlon_75_100_PN_ini, 'DUR_50_75_PN_INI':Dlon_50_75_PN_ini, 'DUR_25_50_PN_INI':Dlon_25_50_PN_ini}
DUR_FIN = {'DUR_75_100_TT_FIN':Dlon_75_100_TT_fin, 'DUR_50_75_TT_FIN':Dlon_50_75_TT_fin, 'DUR_25_50_TT_FIN':Dlon_25_50_TT_fin, 'DUR_75_100_PP_FIN':Dlon_75_100_PP_fin, 'DUR_50_75_PP_FIN':Dlon_50_75_PP_fin, 'DUR_25_50_PP_FIN':Dlon_25_50_PP_fin, 'DUR_75_100_PN_FIN':Dlon_75_100_PN_fin, 'DUR_50_75_PN_FIN':Dlon_50_75_PN_fin, 'DUR_25_50_PN_FIN':Dlon_25_50_PN_fin}






#[-0.125, 20.125, -100.125, -74.875] # = [313, 394, 1039, 1140]
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/2000/CCMP_Wind_Analysis_20000101_V02.0_L3.0_RSS.nc')
Lat     = archivo['latitude'][313:395]
Lon     = archivo['longitude'][1039:1141]-360


def plotear_vientos(chorro, fecha, bar_min, bar_max, comp_u, comp_v, perc, criterio, path): 
	#chorro      : string con el nombre del chorro en cuestion
	#fecha       : string con fecha del mapa que se está ploteando
	#bar_min     : float del límite inferior de la barra de colores
	#bar_max     : float del límite superior de la barra de colores
	#comp_u      : matriz de numpy de la componente u del mapa a plotearse
	#comp_v      : matriz de numpy de la componente v del mapa a plotearse
	#perc        : string que dice el percentil que se está ploteando. Puede ser '75_100', '50_75' o '25_50'
	#criterio    : string "Duracion", "Max" o "Mean"
	#path        : string de directorio donde se guardarán imágenes
	

	#return                 : gráfica o mapa  

	mapa = np.sqrt(comp_u *comp_u + comp_v*comp_v)
	fig  = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
	ax   = fig.add_axes([0.1,0.1,0.8,0.8])

	map = Basemap(projection='merc', llcrnrlat=-0.125, urcrnrlat=20.125, llcrnrlon=-100.125, urcrnrlon=-74.875, resolution='i')
	map.drawcoastlines(linewidth = 0.8)
	map.drawcountries(linewidth = 0.8)
	map.drawparallels(np.arange(0, 20, 50), labels=[1,0,0,1])
	map.drawmeridians(np.arange(-75, -100, 5), labels=[1,0,0,1])

	lons,lats = np.meshgrid(Lon,Lat)
	x,y = map(lons,lats)

	bounds = np.linspace(bar_min, bar_max, 20)
	bounds = np.around(bounds, decimals=2) 


	CF1 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
	CF2 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r

	cb1 = plt.colorbar(CF1, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cb1.set_label('m/s')
	ax.set_title(fecha + ' - ' +chorro+ ' (' + criterio + ' ' + perc + '%)', size='15')
	

	Q = map.quiver(x[::2,::2], y[::2,::2], comp_u[::2,::2], comp_v[::2,::2], scale=150)
	plt.quiverkey(Q, 0.93, 0.05, 10, '10 m/s' )

	map.fillcontinents(color='white')
	plt.savefig(path+fecha+'.png', bbox_inches='tight', dpi=300)
	plt.close()


def plotea_eventos(chorro, criterio, fechas_inicio, fechas_final, perc):
	#chorro        : string con el nombre del chorro en cuestion
	#criterio      : 'MAX', 'MEAN', 'DURACION'
	#fechas_inicio : lista que contiene strings con fechas en que empiezan ciertos eventos. Contiene la misma cantidad que fechas_final
	#fechas_final  : lista que contiene strings con fechas en que terminan ciertos eventos. Contiene la misma cantidad que fechas_inicio
	#perc          : string que dice el percentil que se está ploteando. Puede ser '75_100', '50_75' o '25_50'

	for i in range(len(fechas_inicio)):

		dates_evento = pd.date_range(pd.Timestamp(fechas_inicio[i])-relativedelta(days=3), pd.Timestamp(fechas_final[i])+relativedelta(days=3), freq='6H')

		U_event = np.zeros((len(dates_evento), 82, 102))
		V_event = np.zeros((len(dates_evento), 82, 102))
		for j, date in enumerate(dates_evento):
			Y = str(date.year)
			M = '%02d' % (date.month,)
			D = '%02d' % (date.day,)

			archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/'+Y+'/CCMP_Wind_Analysis_'+Y+M+D+'_V02.0_L3.0_RSS.nc')
			U_event[j] = archivo['uwnd'][date.hour/6, 313:395, 1039:1141]
			V_event[j] = archivo['vwnd'][date.hour/6, 313:395, 1039:1141]

		bar_min = np.min(np.sqrt(U_event*U_event + V_event*V_event))
		bar_max = np.max(np.sqrt(U_event*U_event + V_event*V_event))

		for k, D in enumerate(dates_evento):
			print D
			path = u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/4_expo_2017/MAPAS_VIENTO/'+chorro+'/'+criterio+'/'+perc+'/'+fechas_inicio[i]+'/'
			plotear_vientos(chorro, str(D), bar_min, bar_max, U_event[k], V_event[k], perc, criterio, path)


for ch in ['TT', 'PP', 'PN']:

	for cri in ['DURACION']:

		for perc in ['25_50']:


			if cri == 'MAX':
				FECHAS_INICIO = MAX_INI[cri+'_'+perc+'_'+ch+'_INI'] 
				FECHAS_FIN    = MAX_FIN[cri+'_'+perc+'_'+ch+'_FIN']

			elif cri == 'DURACION':
				FECHAS_INICIO = DUR_INI['DUR_'+perc+'_'+ch+'_INI'] 
				FECHAS_FIN    = DUR_FIN['DUR_'+perc+'_'+ch+'_FIN']


		print ch, cri, perc
		plotea_eventos(ch, cri, FECHAS_INICIO, FECHAS_FIN, perc)
		





