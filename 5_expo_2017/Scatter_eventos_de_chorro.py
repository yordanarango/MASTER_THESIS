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

#==============================================================================
'Calculo máximo descenso de temperatura'
#==============================================================================

def max_desc(serie):
    ref = serie[0]

    dif  = ref - serie

    desc = np.min(dif)
    if desc >= 0:
        desc = np.NAN

    return -1 * desc

desc_temp_sup_TT = np.zeros(len(EVN_TT))
desc_temp_sup_PP = np.zeros(len(EVN_PP))
desc_temp_sup_PN = np.zeros(len(EVN_PN))

for i, ini, fin in zip(range(len(EVN_TT)), Dates_TT_ini, Dates_TT_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_sup_TT[i] = max_desc(data_TT[INI:FIN+1, 0])

for i, ini, fin in zip(range(len(EVN_PP)), Dates_PP_ini, Dates_PP_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_sup_PP[i] = max_desc(data_PP[INI:FIN+1, 0])

for i, ini, fin in zip(range(len(EVN_PN)), Dates_PN_ini, Dates_PN_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_sup_PN[i] = max_desc(data_PN[INI:FIN+1, 0])


desc_temp_10m_TT = np.zeros(len(EVN_TT))
desc_temp_10m_PP = np.zeros(len(EVN_PP))
desc_temp_10m_PN = np.zeros(len(EVN_PN))

for i, ini, fin in zip(range(len(EVN_TT)), Dates_TT_ini, Dates_TT_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_10m_TT[i] = max_desc(data_TT[INI:FIN+1, 7])

for i, ini, fin in zip(range(len(EVN_PP)), Dates_PP_ini, Dates_PP_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_10m_PP[i] = max_desc(data_PP[INI:FIN+1, 7])

for i, ini, fin in zip(range(len(EVN_PN)), Dates_PN_ini, Dates_PN_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_10m_PN[i] = max_desc(data_PN[INI:FIN+1, 7])


desc_temp_19m_TT = np.zeros(len(EVN_TT))
desc_temp_19m_PP = np.zeros(len(EVN_PP))
desc_temp_19m_PN = np.zeros(len(EVN_PN))

for i, ini, fin in zip(range(len(EVN_TT)), Dates_TT_ini, Dates_TT_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_19m_TT[i] = max_desc(data_TT[INI:FIN+1, 11])

for i, ini, fin in zip(range(len(EVN_PP)), Dates_PP_ini, Dates_PP_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_19m_PP[i] = max_desc(data_PP[INI:FIN+1, 11])

for i, ini, fin in zip(range(len(EVN_PN)), Dates_PN_ini, Dates_PN_fin):
    INI = np.where(fechas == dt.datetime.strptime(ini[:10], '%Y-%m-%d')-relativedelta(days=3))[0][0]
    FIN = np.where(fechas == dt.datetime.strptime(fin[:10], '%Y-%m-%d')+relativedelta(days=3))[0][0]
    desc_temp_19m_PN[i] = max_desc(data_PN[INI:FIN+1, 11])


"Colores"
rojos     = ['#FF5733', '#e74c3c', '#a93226']
azules    = ['#5dade2', '#2980b9', '#1f618d']
grises    = ['#abb2b9', '#839192', '#5d6d7e']
amarillos = ['#f4d03f', '#d4ac0d', '#7d6608']
verdes    = ['#7dcea0', '#28b463', '#196f3d']
naranjas  = ['#e59866', '#d35400', '#a04000']
morados   = ['#a569bd', '#7d3c98', '#5b2c6f']
grey      = ['#99a3a4', '#707b7c', '#515a5a']
aguamar   = ['#5dade2', '#2874a6', '#1b4f72']

def plot_scatter(var1, var2, namevar1, namevar2, undsvar1, undsvar2, namejet, Color, path):
    plt.figure(figsize=(8,6), edgecolor='W', facecolor='W')
    plt.scatter(var1, var2, marker=".", color=Color)
    plt.grid(True)
    plt.ylabel(namevar2+'('+undsvar2+')')
    plt.xlabel(namevar1+'('+undsvar1+')')
    plt.title(namevar1+' vs '+namevar2+' ('+namejet+')')
    plt.savefig(path+namejet+'/'+namevar1+'_vs_'+namevar2+'.png',dpi=100,bbox_inches='tight')

path = u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/5_expo_2017/scatter_caracteristicas_chorros/'

CH   = 'TT'
# plot_scatter(lon_TT / 4. , max_TT, 'Dur', 'Max', u'days', 'm/s', CH, rojos[0], path)
# plot_scatter(lon_TT / 4., mean_TT, 'Dur', 'Mean', u'days', 'm/s', CH, azules[0], path)
# plot_scatter(mean_TT, max_TT, 'Mean', 'Max', u'm/s', 'm/s', CH, grises[0], path)
plot_scatter(mean_TT, desc_temp_sup_TT, 'Mean', u'Temperature Decrease (Sup)', 'm/s', u'°C', CH, amarillos[0], path)
plot_scatter(lon_TT / 4., desc_temp_sup_TT, 'Dur', u'Temperature Decrease (Sup)', 'days', u'°C', CH, verdes[0], path)
plot_scatter(mean_TT, desc_temp_10m_TT, 'Mean', u'Temperature Decrease (10m)', 'm/s', u'°C', CH, naranjas[0], path)
plot_scatter(lon_TT / 4., desc_temp_10m_TT, 'Dur', u'Temperature Decrease (10m)', 'days', u'°C', CH, morados[0], path)
plot_scatter(mean_TT, desc_temp_19m_TT, 'Mean', u'Temperature Decrease (19m)', 'm/s', u'°C', CH, grey[0], path)
plot_scatter(lon_TT / 4., desc_temp_19m_TT, 'Dur', u'Temperature Decrease (19m)', 'days', u'°C', CH, aguamar[0], path)

CH   = 'PP'
# plot_scatter(lon_PP / 4. , max_PP, 'Dur', 'Max', u'days', 'm/s', CH, rojos[1], path)
# plot_scatter(lon_PP / 4., mean_PP, 'Dur', 'Mean', u'days', 'm/s', CH, azules[1], path)
# plot_scatter(mean_PP, max_PP, 'Mean', 'Max', u'm/s', 'm/s', CH, grises[1], path)
plot_scatter(mean_PP, desc_temp_sup_PP, 'Mean', u'Temperature Decrease (Sup)', 'm/s', u'°C', CH, amarillos[0], path)
plot_scatter(lon_PP / 4., desc_temp_sup_PP, 'Dur', u'Temperature Decrease (Sup)', 'days', u'°C', CH, verdes[0], path)
plot_scatter(mean_PP, desc_temp_10m_PP, 'Mean', u'Temperature Decrease (10m)', 'm/s', u'°C', CH, naranjas[0], path)
plot_scatter(lon_PP / 4., desc_temp_10m_PP, 'Dur', u'Temperature Decrease (10m)', 'days', u'°C', CH, morados[0], path)
plot_scatter(mean_PP, desc_temp_19m_PP, 'Mean', u'Temperature Decrease (19m)', 'm/s', u'°C', CH, grey[0], path)
plot_scatter(lon_PP / 4., desc_temp_19m_PP, 'Dur', u'Temperature Decrease (19m)', 'days', u'°C', CH, aguamar[0], path)

CH   = 'PN'
# plot_scatter(lon_PN / 4. , max_PN, 'Dur', 'Max', u'days', 'm/s', CH, rojos[2], path)
# plot_scatter(lon_PN / 4., mean_PN, 'Dur', 'Mean', u'days', 'm/s', CH, azules[2], path)
# plot_scatter(mean_PN, max_PN, 'Mean', 'Max', u'm/s', 'm/s', CH, grises[2], path)
plot_scatter(mean_PN, desc_temp_sup_PN, 'Mean', u'Temperature Decrease (Sup)', 'm/s', u'°C', CH, amarillos[0], path)
plot_scatter(lon_PN / 4., desc_temp_sup_PN, 'Dur', u'Temperature Decrease (Sup)', 'days', u'°C', CH, verdes[0], path)
plot_scatter(mean_PN, desc_temp_10m_PN, 'Mean', u'Temperature Decrease (10m)', 'm/s', u'°C', CH, naranjas[0], path)
plot_scatter(lon_PN / 4., desc_temp_10m_PN, 'Dur', u'Temperature Decrease (10m)', 'days', u'°C', CH, morados[0], path)
plot_scatter(mean_PN, desc_temp_19m_PN, 'Mean', u'Temperature Decrease (19m)', 'm/s', u'°C', CH, grey[0], path)
plot_scatter(lon_PN / 4., desc_temp_19m_PN, 'Dur', u'Temperature Decrease (19m)', 'days', u'°C', CH, aguamar[0], path)



