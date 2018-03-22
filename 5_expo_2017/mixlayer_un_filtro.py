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
import pandas as pd
import pickle
import xlrd
import matplotlib.colors as colors
import datetime as dt
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

#==============================================================================
"Tomando los datos de ST"
#==============================================================================

fechas = pd.date_range('1993-01-01', '2015-12-29', freq='D')
data_TT = np.zeros((len(fechas), 24))
data_PP = np.zeros((len(fechas), 24))
data_PN = np.zeros((len(fechas), 24))
for i in range(1993,2016):  
    ArchivoTT = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/SSH_y_ST_GLORYS/puntual/Tehuantepec_puntual/'+str(i)+'.nc') #TT
    ArchivoPP = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/SSH_y_ST_GLORYS/puntual/Papagayos_puntual/'+str(i)+'.nc') #PP     
    ArchivoPN = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/SSH_y_ST_GLORYS/puntual/Panama_puntual/'+str(i)+'.nc') #PN   
    time = ArchivoPP.variables['time'][:]    
    cdftime = utime('hours since 1950-01-01 00:00:00', calendar='gregorian')
    date = [cdftime.num2date(k) for k in time]
    DATE = pd.date_range(date[0], date[1], freq='D')
    
    fecha_inicio = pd.date_range('1993-01-01', str(i)+'-01-01', freq='D')
    fecha_final  = pd.date_range('1993-01-01', str(i)+'-12-31', freq='D')
    
    if i == 2015:
        data_TT[len(fecha_inicio)-1:]                 = ArchivoTT.variables['temperature'][:,:,1,1]
        data_PP[len(fecha_inicio)-1:]                 = ArchivoPP.variables['temperature'][:,:,1,1]
        data_PN[len(fecha_inicio)-1:]                 = ArchivoPN.variables['temperature'][:,:,1,1]
    else:    
        data_TT[len(fecha_inicio)-1:len(fecha_final)] = ArchivoTT.variables['temperature'][:,:,1,1]
        data_PP[len(fecha_inicio)-1:len(fecha_final)] = ArchivoPP.variables['temperature'][:,:,1,1]
        data_PN[len(fecha_inicio)-1:len(fecha_final)] = ArchivoPN.variables['temperature'][:,:,1,1]

data_TT = data_TT-273.15
data_PP = data_PP-273.15
data_PN = data_PN-273.15

#==============================================================================
"Calculando ciclo anual"
#==============================================================================
ciclo_anual_TT = np.zeros((12, 24))
ciclo_anual_PP = np.zeros((12, 24))
ciclo_anual_PN = np.zeros((12, 24))

for i in range(24):
    for j in range(12):
        ciclo_anual_TT[j, i] = np.mean(data_TT[(fechas.month == j+1),i])
        ciclo_anual_PP[j, i] = np.mean(data_PP[(fechas.month == j+1),i])
        ciclo_anual_PN[j, i] = np.mean(data_PN[(fechas.month == j+1),i])

#==============================================================================
"Calculando anomalías"
#==============================================================================
anom_TT = np.zeros(data_TT.shape)
anom_PP = np.zeros(data_PP.shape)
anom_PN = np.zeros(data_PN.shape)

for i in range(24):
    for j in range(len(fechas)):
        anom_TT[j, i] = data_TT[j, i] - ciclo_anual_TT[fechas[j].month-1, i]
        anom_PP[j, i] = data_PP[j, i] - ciclo_anual_PP[fechas[j].month-1, i]
        anom_PN[j, i] = data_PN[j, i] - ciclo_anual_PN[fechas[j].month-1, i]

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

Dates_TT_max = [str(Dates_RASI[(x[0] + np.where(RASI_Tehuantepec[x[0]:x[1]+1] == np.max(RASI_Tehuantepec[x[0]:x[1]+1])))[0][0]]) for x in EVN_TT] #fecha en que se da el máximo de cada evento
Dates_PP_max = [str(Dates_RASI[(x[0] + np.where(RASI_Papagayo[x[0]:x[1]+1] == np.max(RASI_Papagayo[x[0]:x[1]+1])))[0][0]]) for x in EVN_PP] #fecha en que se da el máximo de cada evento
Dates_PN_max = [str(Dates_RASI[(x[0] + np.where(RASI_Panama[x[0]:x[1]+1] == np.max(RASI_Panama[x[0]:x[1]+1])))[0][0]]) for x in EVN_PN] #fecha en que se da el máximo de cada evento

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

caract_TT                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin', 'Fechas max'])
caract_TT['Duracion']      = lon_TT
caract_TT['Max']           = max_TT
caract_TT['Mean']          = mean_TT
caract_TT['Fechas inicio'] = Dates_TT_ini
caract_TT['Fechas fin']    = Dates_TT_fin
caract_TT['Fechas max']    = Dates_TT_max
sort_by_lon_TT = caract_TT.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_TT = caract_TT.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_TT = caract_TT.sort_values(by = ['Mean'], ascending=[False])
lon_75_TT = np.percentile(lon_TT, 75); lon_50_TT = np.percentile(lon_TT, 50); lon_25_TT = np.percentile(lon_TT, 25)
max_75_TT = np.percentile(max_TT, 75); max_50_TT = np.percentile(max_TT, 50); max_25_TT = np.percentile(max_TT, 25)
mean_75_TT = np.percentile(mean_TT, 75); mean_50_TT = np.percentile(mean_TT, 50); mean_25_TT = np.percentile(mean_TT, 25)


caract_PP                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin', 'Fechas max'])
caract_PP['Duracion']      = lon_PP
caract_PP['Max']           = max_PP
caract_PP['Mean']          = mean_PP
caract_PP['Fechas inicio'] = Dates_PP_ini
caract_PP['Fechas fin']    = Dates_PP_fin
caract_PP['Fechas max']    = Dates_PP_max
sort_by_lon_PP = caract_PP.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_PP = caract_PP.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_PP = caract_PP.sort_values(by = ['Mean'], ascending=[False])
lon_75_PP = np.percentile(lon_PP, 75); lon_50_PP = np.percentile(lon_PP, 50); lon_25_PP = np.percentile(lon_PP, 25)
max_75_PP = np.percentile(max_PP, 75); max_50_PP = np.percentile(max_PP, 50); max_25_PP = np.percentile(max_PP, 25)
mean_75_PP = np.percentile(mean_PP, 75); mean_50_PP = np.percentile(mean_PP, 50); mean_25_PP = np.percentile(mean_PP, 25)


caract_PN                  = pd.DataFrame(columns = ['Duracion', 'Max', 'Mean', 'Fechas inicio', 'Fechas fin', 'Fechas max'])
caract_PN['Duracion']      = lon_PN
caract_PN['Max']           = max_PN
caract_PN['Mean']          = mean_PN
caract_PN['Fechas inicio'] = Dates_PN_ini
caract_PN['Fechas fin']    = Dates_PN_fin
caract_PN['Fechas max']    = Dates_PN_max
sort_by_lon_PN = caract_PN.sort_values(by = ['Duracion'], ascending=[False])
sort_by_max_PN = caract_PN.sort_values(by = ['Max'], ascending=[False])
sort_by_mean_PN = caract_PN.sort_values(by = ['Mean'], ascending=[False])
lon_75_PN = np.percentile(lon_PN, 75); lon_50_PN = np.percentile(lon_PN, 50); lon_25_PN = np.percentile(lon_PN, 25)
max_75_PN = np.percentile(max_PN, 75); max_50_PN = np.percentile(max_PN, 50); max_25_PN = np.percentile(max_PN, 25)
mean_75_PN = np.percentile(mean_PN, 75); mean_50_PN = np.percentile(mean_PN, 50); mean_25_PN = np.percentile(mean_PN, 25)


N = 100


#
"Fechas max"
#

Dmax_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_TT_max = caract_TT['Fechas max'][caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_TT_max  = caract_TT['Fechas max'][caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_TT_max  = caract_TT['Fechas max'][caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:N]].values


Dmax_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_PP_max = caract_PP['Fechas max'][caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PP_max  = caract_PP['Fechas max'][caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PP_max  = caract_PP['Fechas max'][caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:N]].values


Dmax_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmax_75_100_PN_max = caract_PN['Fechas max'][caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmax_50_75_PN_max  = caract_PN['Fechas max'][caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmax_25_50_PN_max  = caract_PN['Fechas max'][caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:N]].values


#
"Fechas mean"
#

Dmean_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_TT_max = caract_TT['Fechas max'][caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_TT_max  = caract_TT['Fechas max'][caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_TT_max  = caract_TT['Fechas max'][caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:N]].values


Dmean_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_PP_max = caract_PP['Fechas max'][caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PP_max  = caract_PP['Fechas max'][caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PP_max  = caract_PP['Fechas max'][caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:N]].values


Dmean_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmean_75_100_PN_max = caract_PN['Fechas max'][caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmean_50_75_PN_max  = caract_PN['Fechas max'][caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dmean_25_50_PN_max  = caract_PN['Fechas max'][caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:N]].values


#
"Fechas Duracion"
#

Dlon_75_100_TT_ini = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_TT_fin = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_TT_max = caract_TT['Fechas max'][caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_TT_max  = caract_TT['Fechas max'][caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_TT_ini  = caract_TT['Fechas inicio'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_TT_fin  = caract_TT['Fechas fin'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_TT_max  = caract_TT['Fechas max'][caract_TT.Duracion[np.nonzero((lon_50_TT >= caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:N]].values


Dlon_75_100_PP_ini = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_PP_fin = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_PP_max = caract_PP['Fechas max'][caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PP_max  = caract_PP['Fechas max'][caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PP_ini  = caract_PP['Fechas inicio'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PP_fin  = caract_PP['Fechas fin'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PP_max  = caract_PP['Fechas max'][caract_PP.Duracion[np.nonzero((lon_50_PP >= caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:N]].values


Dlon_75_100_PN_ini = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_PN_fin = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dlon_75_100_PN_max = caract_PN['Fechas max'][caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dlon_50_75_PN_max  = caract_PN['Fechas max'][caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PN_ini  = caract_PN['Fechas inicio'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PN_fin  = caract_PN['Fechas fin'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:N]].values
Dlon_25_50_PN_max  = caract_PN['Fechas max'][caract_PN.Duracion[np.nonzero((lon_50_PN >= caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:N]].values

#
"Diccionarios con fechas"
#
MAX_INI = {'MAX_75_100_TT_INI':Dmax_75_100_TT_ini, 'MAX_50_75_TT_INI':Dmax_50_75_TT_ini, 'MAX_25_50_TT_INI':Dmax_25_50_TT_ini, 'MAX_75_100_PP_INI':Dmax_75_100_PP_ini, 'MAX_50_75_PP_INI':Dmax_50_75_PP_ini, 'MAX_25_50_PP_INI':Dmax_25_50_PP_ini, 'MAX_75_100_PN_INI':Dmax_75_100_PN_ini, 'MAX_50_75_PN_INI':Dmax_50_75_PN_ini, 'MAX_25_50_PN_INI':Dmax_25_50_PN_ini}
MAX_FIN = {'MAX_75_100_TT_FIN':Dmax_75_100_TT_fin, 'MAX_50_75_TT_FIN':Dmax_50_75_TT_fin, 'MAX_25_50_TT_FIN':Dmax_25_50_TT_fin, 'MAX_75_100_PP_FIN':Dmax_75_100_PP_fin, 'MAX_50_75_PP_FIN':Dmax_50_75_PP_fin, 'MAX_25_50_PP_FIN':Dmax_25_50_PP_fin, 'MAX_75_100_PN_FIN':Dmax_75_100_PN_fin, 'MAX_50_75_PN_FIN':Dmax_50_75_PN_fin, 'MAX_25_50_PN_FIN':Dmax_25_50_PN_fin}
MAX_MAX = {'MAX_75_100_TT_MAX':Dmax_75_100_TT_max, 'MAX_50_75_TT_MAX':Dmax_50_75_TT_max, 'MAX_25_50_TT_MAX':Dmax_25_50_TT_max, 'MAX_75_100_PP_MAX':Dmax_75_100_PP_max, 'MAX_50_75_PP_MAX':Dmax_50_75_PP_max, 'MAX_25_50_PP_MAX':Dmax_25_50_PP_max, 'MAX_75_100_PN_MAX':Dmax_75_100_PN_max, 'MAX_50_75_PN_MAX':Dmax_50_75_PN_max, 'MAX_25_50_PN_MAX':Dmax_25_50_PN_max}

MEAN_INI = {'MEAN_75_100_TT_INI':Dmean_75_100_TT_ini, 'MEAN_50_75_TT_INI':Dmean_50_75_TT_ini, 'MEAN_25_50_TT_INI':Dmean_25_50_TT_ini, 'MEAN_75_100_PP_INI':Dmean_75_100_PP_ini, 'MEAN_50_75_PP_INI':Dmean_50_75_PP_ini, 'MEAN_25_50_PP_INI':Dmean_25_50_PP_ini, 'MEAN_75_100_PN_INI':Dmean_75_100_PN_ini, 'MEAN_50_75_PN_INI':Dmean_50_75_PN_ini, 'MEAN_25_50_PN_INI':Dmean_25_50_PN_ini}
MEAN_FIN = {'MEAN_75_100_TT_FIN':Dmean_75_100_TT_fin, 'MEAN_50_75_TT_FIN':Dmean_50_75_TT_fin, 'MEAN_25_50_TT_FIN':Dmean_25_50_TT_fin, 'MEAN_75_100_PP_FIN':Dmean_75_100_PP_fin, 'MEAN_50_75_PP_FIN':Dmean_50_75_PP_fin, 'MEAN_25_50_PP_FIN':Dmean_25_50_PP_fin, 'MEAN_75_100_PN_FIN':Dmean_75_100_PN_fin, 'MEAN_50_75_PN_FIN':Dmean_50_75_PN_fin, 'MEAN_25_50_PN_FIN':Dmean_25_50_PN_fin}
MEAN_MAX = {'MEAN_75_100_TT_MAX':Dmean_75_100_TT_max, 'MEAN_50_75_TT_MAX':Dmean_50_75_TT_max, 'MEAN_25_50_TT_MAX':Dmean_25_50_TT_max, 'MEAN_75_100_PP_MAX':Dmean_75_100_PP_max, 'MEAN_50_75_PP_MAX':Dmean_50_75_PP_max, 'MEAN_25_50_PP_MAX':Dmean_25_50_PP_max, 'MEAN_75_100_PN_MAX':Dmean_75_100_PN_max, 'MEAN_50_75_PN_MAX':Dmean_50_75_PN_max, 'MEAN_25_50_PN_MAX':Dmean_25_50_PN_max}

DUR_INI = {'DUR_75_100_TT_INI':Dlon_75_100_TT_ini, 'DUR_50_75_TT_INI':Dlon_50_75_TT_ini, 'DUR_25_50_TT_INI':Dlon_25_50_TT_ini, 'DUR_75_100_PP_INI':Dlon_75_100_PP_ini, 'DUR_50_75_PP_INI':Dlon_50_75_PP_ini, 'DUR_25_50_PP_INI':Dlon_25_50_PP_ini, 'DUR_75_100_PN_INI':Dlon_75_100_PN_ini, 'DUR_50_75_PN_INI':Dlon_50_75_PN_ini, 'DUR_25_50_PN_INI':Dlon_25_50_PN_ini}
DUR_FIN = {'DUR_75_100_TT_FIN':Dlon_75_100_TT_fin, 'DUR_50_75_TT_FIN':Dlon_50_75_TT_fin, 'DUR_25_50_TT_FIN':Dlon_25_50_TT_fin, 'DUR_75_100_PP_FIN':Dlon_75_100_PP_fin, 'DUR_50_75_PP_FIN':Dlon_50_75_PP_fin, 'DUR_25_50_PP_FIN':Dlon_25_50_PP_fin, 'DUR_75_100_PN_FIN':Dlon_75_100_PN_fin, 'DUR_50_75_PN_FIN':Dlon_50_75_PN_fin, 'DUR_25_50_PN_FIN':Dlon_25_50_PN_fin}
DUR_MAX = {'DUR_75_100_TT_MAX':Dlon_75_100_TT_max, 'DUR_50_75_TT_MAX':Dlon_50_75_TT_max, 'DUR_25_50_TT_MAX':Dlon_25_50_TT_max, 'DUR_75_100_PP_MAX':Dlon_75_100_PP_max, 'DUR_50_75_PP_MAX':Dlon_50_75_PP_max, 'DUR_25_50_PP_MAX':Dlon_25_50_PP_max, 'DUR_75_100_PN_MAX':Dlon_75_100_PN_max, 'DUR_50_75_PN_MAX':Dlon_50_75_PN_max, 'DUR_25_50_PN_MAX':Dlon_25_50_PN_max}


#==============================================================================
'Encuentra profundidad capa de mezcla'
#==============================================================================

def mld(temp, prof, fechas, DELT = 0.2):
    """
    temp : matriz de perfiles de temperatura (o salinidad) con las filas la profundidad y las columnas las fechas
    prof : vector de profundidades de los perfiles modelados
    fechas: fechas de los perfiles
    DELT : delta de temperatura o salinidad para calcular mld
    """
    MLD_mod = []
    po_mod = [int(i) for i in prof]; po_mod = np.array(po_mod)
    to_mod = np.array(temp)
    pref_mod = np.where(po_mod[:] == 9)[0] # esta es la profundidad de referencia, normalmente se elige 10 m
   
    for j in range(np.shape(to_mod)[1]): # Recorre las columnas (fechas de medicion)
        Tref_mod = to_mod[pref_mod,j]
        Tb_mod = Tref_mod - DELT
        dif_mod = map(lambda x: np.abs(x-Tb_mod), to_mod[:,j])    
        R_mod = np.array([dif_mod,to_mod[:,j],po_mod]).T
        dif_min_mod = np.array (sorted(R_mod, key=lambda r_mod: r_mod[0]))
        Ti_mod = dif_min_mod[0,1]
        Pi_mod = dif_min_mod[0,2] # profundidad de la capa de mezcla del perfil i
        MLD_mod.append(Pi_mod)

    mld_pandas = pd.Series(data=MLD_mod, index=fechas)
    return mld_pandas


#==============================================================================
'Ploteando profundidad capa de mezcla'
#==============================================================================

depth = ArchivoPN['depth'][:]
path  = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/5_expo_2017/mix_layer_depth/'
for ch in ['TT', 'PP', 'PN']:

    for cri in ['MEAN']:

        for perc in ['75_100']:


            if cri == 'MAX':
                FECHAS_INICIO = MAX_INI[cri+'_'+perc+'_'+ch+'_INI'] 
                FECHAS_FIN    = MAX_FIN[cri+'_'+perc+'_'+ch+'_FIN']
                FECHAS_MAX    = MAX_MAX[cri+'_'+perc+'_'+ch+'_MAX']

            elif cri == 'MEAN':
                FECHAS_INICIO = MEAN_INI[cri+'_'+perc+'_'+ch+'_INI'] 
                FECHAS_FIN    = MEAN_FIN[cri+'_'+perc+'_'+ch+'_FIN']
                FECHAS_MAX    = MEAN_MAX[cri+'_'+perc+'_'+ch+'_MAX']

            elif cri == 'DURACION':
                FECHAS_INICIO = DUR_INI['DUR_'+perc+'_'+ch+'_INI'] 
                FECHAS_FIN    = DUR_FIN['DUR_'+perc+'_'+ch+'_FIN']
                FECHAS_MAX    = DUR_MAX['DUR_'+perc+'_'+ch+'_MAX']

            fig = plt.figure(figsize=(8,6), edgecolor='W',facecolor='W')
            ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
            
            for c, f, m in zip(FECHAS_INICIO, FECHAS_FIN, FECHAS_MAX):

                print ch, cri, perc
                date_evento = pd.date_range(dt.datetime.strptime(c[:10],'%Y-%m-%d')-relativedelta(days=2), dt.datetime.strptime(f[:10],'%Y-%m-%d')+relativedelta(days=3), freq='D')
                n_back_max  = len(pd.date_range(dt.datetime.strptime(c[:10],'%Y-%m-%d')-relativedelta(days=2), dt.datetime.strptime(m[:10],'%Y-%m-%d'), freq='D')) - 1
                C           = np.where(fechas == pd.Timestamp(c[:10])-relativedelta(days=2))[0][0] #Comienzo
                F           = np.where(fechas == pd.Timestamp(f[:10])+relativedelta(days=3))[0][0] #Final

                if   ch == 'TT': ST = data_TT[C:F+1].T
                elif ch == 'PP': ST = data_PP[C:F+1].T
                elif ch == 'PN': ST = data_PN[C:F+1].T

                PCM      = np.array(mld(ST[:-3,:], depth[:-3], date_evento, DELT = 1.0).values)
                vector_x = np.arange(-n_back_max, -n_back_max + len(date_evento))

                #if len(PCM) == 1 or len(PCM) == 2:
                ax1.plot(vector_x, PCM)
                #else:
                #    ax1.plot(vector_x[:2],   PCM[:2],   color='b')
                #    ax1.plot(vector_x[2:-3], PCM[2:-3], color='k')
                #    ax1.plot(vector_x[-3:],  PCM[-3:],  color='r')

            ax1.set_title(ch + ' - (' + cri + ' ' + perc + '%)')
            ax1.invert_yaxis()
            ax1.axvline(x=0.0, color='k', alpha=0.7)
            #ax1.set_xlim(11, 30)
            ax1.set_ylabel(u'Mix Layer Depth (m)')
            ax1.set_xlabel(u'Time (Days) ')
            plt.grid(True)            
            plt.savefig(path+ch+'_'+cri+'_'+perc+'.png',dpi=100,bbox_inches='tight')




