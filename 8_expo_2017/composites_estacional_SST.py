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
import os

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
	for i, mes in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']):
		for j, hora in enumerate(['0', '6', '12', '18']):
			pos    = np.where((fechas.month == i+1 ) & (fechas.hour == int(hora)))[0]
			M      = np.zeros((len(pos), len_lat, len_lon))

			for k, l in enumerate(pos):
				M[k] = matriz[l]

			media = np.mean(M, axis=0)

			Dict_ciclo.update({mes+'_'+hora:media})

	return Dict_ciclo


def plotear(lllat, urlat, lllon, urlon, dist_lat, dist_lon, Lon, Lat, mapa, bar_min, bar_max, unds, titulo, path, C_T='k', wind=False, mapa_u=None, mapa_v=None):

	# lllat (low-left-lat)   : float con la latitud de la esquina inferior izquierda
	# urlat (uper-right-lat) : float con la latitud de la esquina superior derecha
	# lllon (low-left-lon)   : float con la longitud de la esquina inferior izquierda en coordenas negativas este
	# urlon (uper-right-lon) : float con la longitud de la esquina superior derecha en coordenas negativas este
	# dist_lat               : entero que indica cada cuanto dibujar las lineas de los paralelos
	# dist_lon               : entero que indica cada cuanto dibujar las lineas de los meridianos
	# Lon                    : array de numpy con las longitudes del mapa a plotearse en coordenadas negativas este
	# Lat                    : array de numpy con las longitudes del mapa a plotearse
	# mapa                   : array de numpy 2D a plotearse con contourf
	# bar_min                : mínimo valor del mapa a plotearse
	# bar_max                : máximo valor del mapa a plotearse
	# unds                   : string de las unidades de la variable que se va a plotear
	# titulo                 : string del titulo que llevará el mapa
	# path                   : 'string de la dirección y el nombre del archivo que contendrá la figura a generarse'
	# wind                   : boolean que diga si se quiere pintar flechas de viento (corrientes), donde True es que sí se va a hacer
	# mapa_u                 : array de numpay 2D con la componente en x de la velocidad y que será utilizado para pintar las flechas. Este tomara algun valor siempre y cuando wind=True
	# mapa_v                 : array de numpay 2D con la componente en y de la velocidad y que será utilizado para pintar las flechas. Este tomara algun valor siempre y cuando wind=True


	# return                 : gráfica o mapa

	fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
	ax = fig.add_axes([0.1,0.1,0.8,0.8])

	map = Basemap(projection='merc', llcrnrlat=lllat, urcrnrlat=urlat, llcrnrlon=lllon, urcrnrlon=urlon, resolution='i')
	map.drawcoastlines(linewidth = 0.8)
	map.drawcountries(linewidth = 0.8)
	map.drawparallels(np.arange(lllat, urlat, dist_lat), labels=[1,0,0,1])
	map.drawmeridians(np.arange(lllon, urlon, dist_lon), labels=[1,0,0,1])

	lons,lats = np.meshgrid(Lon,Lat)
	x,y = map(lons,lats)

	bounds = np.linspace(bar_min, bar_max, 20)
	bounds = np.around(bounds, decimals=2)

	if wind == False:
		CF1 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.RdYlBu_r, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.RdYlBu_r, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r
	else:
		CF1 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r

	cb1 = plt.colorbar(CF1, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cb1.set_label(unds)
	ax.set_title(titulo, size='15', color = C_T)

	if wind == True:
		Q = map.quiver(x[::2,::2], y[::2,::2], mapa_u[::2,::2], mapa_v[::2,::2], scale=150)
		plt.quiverkey(Q, 0.93, 0.05, 10, '10 m/s' )

	#map.fillcontinents(color='white')
	plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
	plt.close('all')
##############################################################################################################

#
'DATOS RASI'
#

RASI_Tehuantepec = []
RASI_Papagayo    = []
RASI_Panama      = []

for i in range(1998, 2012):
	mean_TT = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Tehuantepec_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PP = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Papagayo_'+str(i)+'.nc')['WindSpeedMean'][:]
	mean_PN = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Panama_'+str(i)+'.nc')['WindSpeedMean'][:]

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
# sort_by_lon_TT = caract_TT.sort_values(by = ['Duracion'], ascending=[False])
# sort_by_max_TT = caract_TT.sort_values(by = ['Max'], ascending=[False])
# sort_by_mean_TT = caract_TT.sort_values(by = ['Mean'], ascending=[False])
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
# sort_by_lon_PP = caract_PP.sort_values(by = ['Duracion'], ascending=[False])
# sort_by_max_PP = caract_PP.sort_values(by = ['Max'], ascending=[False])
# sort_by_mean_PP = caract_PP.sort_values(by = ['Mean'], ascending=[False])
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
# sort_by_lon_PN = caract_PN.sort_values(by = ['Duracion'], ascending=[False])
# sort_by_max_PN = caract_PN.sort_values(by = ['Max'], ascending=[False])
# sort_by_mean_PN = caract_PN.sort_values(by = ['Mean'], ascending=[False])
lon_75_PN = np.percentile(lon_PN, 75); lon_50_PN = np.percentile(lon_PN, 50); lon_25_PN = np.percentile(lon_PN, 25)
max_75_PN = np.percentile(max_PN, 75); max_50_PN = np.percentile(max_PN, 50); max_25_PN = np.percentile(max_PN, 25)
mean_75_PN = np.percentile(mean_PN, 75); mean_50_PN = np.percentile(mean_PN, 50); mean_25_PN = np.percentile(mean_PN, 25)


#
"FECHAS MAX - FILTRADOS POR MEAN"
#
"TT"
Dmax_75_100_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

"PP"

Dmax_75_100_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

"PN"

Dmax_75_100_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values


#
"FECHAS MAX - FILTRADOS POR DURACION"
#
"TT"
Dmax_75_100_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values


"PP"
Dmax_75_100_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values


"PN"

Dmax_75_100_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_75_100_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_75_100_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_50_75_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_50_75_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmax_25_50_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmax_25_50_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values


#
"FECHAS MEAN - FILTRADOS POR DURACION"
#
"TT"
Dmean_75_100_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:])].values


"PP"
Dmean_75_100_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:])].values

"PN"

Dmean_75_100_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_lon_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_lon_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:])].values


#
"FECHAS MEAN - FILTRADOS POR MAX"
#
"TT"
Dmean_75_100_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values


"PP"
Dmean_75_100_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values


"PN"

Dmean_75_100_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_75_100_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_75_100_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_50_75_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_50_75_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dmean_25_50_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dmean_25_50_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

#
"FECHAS DURACION - FILTRADOS POR MAX"
#
"TT"
Dlon_75_100_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero(caract_TT.Max > max_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_75_TT > caract_TT.Max) & (caract_TT.Max > max_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Max[np.nonzero((max_50_TT > caract_TT.Max) & (caract_TT.Max > max_25_TT))[0]].sort_values(ascending=False).index[:])].values


"PP"
Dlon_75_100_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero(caract_PP.Max > max_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_75_PP > caract_PP.Max) & (caract_PP.Max > max_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Max[np.nonzero((max_50_PP > caract_PP.Max) & (caract_PP.Max > max_25_PP))[0]].sort_values(ascending=False).index[:])].values


"PN"

Dlon_75_100_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_max_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero(caract_PN.Max > max_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_75_PN > caract_PN.Max) & (caract_PN.Max > max_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_max_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Max[np.nonzero((max_50_PN > caract_PN.Max) & (caract_PN.Max > max_25_PN))[0]].sort_values(ascending=False).index[:])].values


#
"FECHAS DUR - FILTRADOS POR MEAN"
#
"TT"
Dlon_75_100_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_TT_ini = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_TT_ini  = caract_TT['Fechas inicio'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_TT_fin = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_TT_fin  = caract_TT['Fechas fin'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero(caract_TT.Duracion > lon_75_TT)[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_75_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_50_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_TT_max = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero(caract_TT.Mean > mean_75_TT)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_75_TT > caract_TT.Mean) & (caract_TT.Mean > mean_50_TT))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_TT_max  = caract_TT['Fechas max'][np.intersect1d(caract_TT.Duracion[np.nonzero((lon_50_TT > caract_TT.Duracion) & (caract_TT.Duracion > lon_25_TT))[0]].sort_values(ascending=False).index[:], caract_TT.Mean[np.nonzero((mean_50_TT > caract_TT.Mean) & (caract_TT.Mean > mean_25_TT))[0]].sort_values(ascending=False).index[:])].values

"PP"

Dlon_75_100_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PP_ini = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PP_ini  = caract_PP['Fechas inicio'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PP_fin = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PP_fin  = caract_PP['Fechas fin'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero(caract_PP.Duracion > lon_75_PP)[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_75_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_50_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PP_max = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero(caract_PP.Mean > mean_75_PP)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_75_PP > caract_PP.Mean) & (caract_PP.Mean > mean_50_PP))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PP_max  = caract_PP['Fechas max'][np.intersect1d(caract_PP.Duracion[np.nonzero((lon_50_PP > caract_PP.Duracion) & (caract_PP.Duracion > lon_25_PP))[0]].sort_values(ascending=False).index[:], caract_PP.Mean[np.nonzero((mean_50_PP > caract_PP.Mean) & (caract_PP.Mean > mean_25_PP))[0]].sort_values(ascending=False).index[:])].values

"PN"

Dlon_75_100_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PN_ini = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PN_ini  = caract_PN['Fechas inicio'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PN_fin = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PN_fin  = caract_PN['Fechas fin'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_75_100_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_75_100_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero(caract_PN.Duracion > lon_75_PN)[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_50_75_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_50_75_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_75_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_50_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values

Dlon_25_50_mean_75_100_PN_max = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero(caract_PN.Mean > mean_75_PN)[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_50_75_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_75_PN > caract_PN.Mean) & (caract_PN.Mean > mean_50_PN))[0]].sort_values(ascending=False).index[:])].values
Dlon_25_50_mean_25_50_PN_max  = caract_PN['Fechas max'][np.intersect1d(caract_PN.Duracion[np.nonzero((lon_50_PN > caract_PN.Duracion) & (caract_PN.Duracion > lon_25_PN))[0]].sort_values(ascending=False).index[:], caract_PN.Mean[np.nonzero((mean_50_PN > caract_PN.Mean) & (caract_PN.Mean > mean_25_PN))[0]].sort_values(ascending=False).index[:])].values



"MAX"

MAX_MEAN_INI = {'MAX_75_100_MEAN_75_100_TT_INI':Dmax_75_100_mean_75_100_TT_ini, 'MAX_75_100_MEAN_50_75_TT_INI':Dmax_75_100_mean_50_75_TT_ini, 'MAX_75_100_MEAN_25_50_TT_INI':Dmax_75_100_mean_25_50_TT_ini, 'MAX_50_75_MEAN_75_100_TT_INI':Dmax_50_75_mean_75_100_TT_ini, 'MAX_50_75_MEAN_50_75_TT_INI':Dmax_50_75_mean_50_75_TT_ini, 'MAX_50_75_MEAN_25_50_TT_INI':Dmax_50_75_mean_25_50_TT_ini, 'MAX_25_50_MEAN_75_100_TT_INI':Dmax_25_50_mean_75_100_TT_ini, 'MAX_25_50_MEAN_50_75_TT_INI':Dmax_25_50_mean_50_75_TT_ini, 'MAX_25_50_MEAN_25_50_TT_INI':Dmax_25_50_mean_25_50_TT_ini,
		   		'MAX_75_100_MEAN_75_100_PP_INI':Dmax_75_100_mean_75_100_PP_ini, 'MAX_75_100_MEAN_50_75_PP_INI':Dmax_75_100_mean_50_75_PP_ini, 'MAX_75_100_MEAN_25_50_PP_INI':Dmax_75_100_mean_25_50_PP_ini, 'MAX_50_75_MEAN_75_100_PP_INI':Dmax_50_75_mean_75_100_PP_ini, 'MAX_50_75_MEAN_50_75_PP_INI':Dmax_50_75_mean_50_75_PP_ini, 'MAX_50_75_MEAN_25_50_PP_INI':Dmax_50_75_mean_25_50_PP_ini, 'MAX_25_50_MEAN_75_100_PP_INI':Dmax_25_50_mean_75_100_PP_ini, 'MAX_25_50_MEAN_50_75_PP_INI':Dmax_25_50_mean_50_75_PP_ini, 'MAX_25_50_MEAN_25_50_PP_INI':Dmax_25_50_mean_25_50_PP_ini,
   		   		'MAX_75_100_MEAN_75_100_PN_INI':Dmax_75_100_mean_75_100_PN_ini, 'MAX_75_100_MEAN_50_75_PN_INI':Dmax_75_100_mean_50_75_PN_ini, 'MAX_75_100_MEAN_25_50_PN_INI':Dmax_75_100_mean_25_50_PN_ini, 'MAX_50_75_MEAN_75_100_PN_INI':Dmax_50_75_mean_75_100_PN_ini, 'MAX_50_75_MEAN_50_75_PN_INI':Dmax_50_75_mean_50_75_PN_ini, 'MAX_50_75_MEAN_25_50_PN_INI':Dmax_50_75_mean_25_50_PN_ini, 'MAX_25_50_MEAN_75_100_PN_INI':Dmax_25_50_mean_75_100_PN_ini, 'MAX_25_50_MEAN_50_75_PN_INI':Dmax_25_50_mean_50_75_PN_ini, 'MAX_25_50_MEAN_25_50_PN_INI':Dmax_25_50_mean_25_50_PN_ini}

MAX_MEAN_FIN = {'MAX_75_100_MEAN_75_100_TT_FIN':Dmax_75_100_mean_75_100_TT_fin, 'MAX_75_100_MEAN_50_75_TT_FIN':Dmax_75_100_mean_50_75_TT_fin, 'MAX_75_100_MEAN_25_50_TT_FIN':Dmax_75_100_mean_25_50_TT_fin, 'MAX_50_75_MEAN_75_100_TT_FIN':Dmax_50_75_mean_75_100_TT_fin, 'MAX_50_75_MEAN_50_75_TT_FIN':Dmax_50_75_mean_50_75_TT_fin, 'MAX_50_75_MEAN_25_50_TT_FIN':Dmax_50_75_mean_25_50_TT_fin, 'MAX_25_50_MEAN_75_100_TT_FIN':Dmax_25_50_mean_75_100_TT_fin, 'MAX_25_50_MEAN_50_75_TT_FIN':Dmax_25_50_mean_50_75_TT_fin, 'MAX_25_50_MEAN_25_50_TT_FIN':Dmax_25_50_mean_25_50_TT_fin,
		   		'MAX_75_100_MEAN_75_100_PP_FIN':Dmax_75_100_mean_75_100_PP_fin, 'MAX_75_100_MEAN_50_75_PP_FIN':Dmax_75_100_mean_50_75_PP_fin, 'MAX_75_100_MEAN_25_50_PP_FIN':Dmax_75_100_mean_25_50_PP_fin, 'MAX_50_75_MEAN_75_100_PP_FIN':Dmax_50_75_mean_75_100_PP_fin, 'MAX_50_75_MEAN_50_75_PP_FIN':Dmax_50_75_mean_50_75_PP_fin, 'MAX_50_75_MEAN_25_50_PP_FIN':Dmax_50_75_mean_25_50_PP_fin, 'MAX_25_50_MEAN_75_100_PP_FIN':Dmax_25_50_mean_75_100_PP_fin, 'MAX_25_50_MEAN_50_75_PP_FIN':Dmax_25_50_mean_50_75_PP_fin, 'MAX_25_50_MEAN_25_50_PP_FIN':Dmax_25_50_mean_25_50_PP_fin,
   		   		'MAX_75_100_MEAN_75_100_PN_FIN':Dmax_75_100_mean_75_100_PN_fin, 'MAX_75_100_MEAN_50_75_PN_FIN':Dmax_75_100_mean_50_75_PN_fin, 'MAX_75_100_MEAN_25_50_PN_FIN':Dmax_75_100_mean_25_50_PN_fin, 'MAX_50_75_MEAN_75_100_PN_FIN':Dmax_50_75_mean_75_100_PN_fin, 'MAX_50_75_MEAN_50_75_PN_FIN':Dmax_50_75_mean_50_75_PN_fin, 'MAX_50_75_MEAN_25_50_PN_FIN':Dmax_50_75_mean_25_50_PN_fin, 'MAX_25_50_MEAN_75_100_PN_FIN':Dmax_25_50_mean_75_100_PN_fin, 'MAX_25_50_MEAN_50_75_PN_FIN':Dmax_25_50_mean_50_75_PN_fin, 'MAX_25_50_MEAN_25_50_PN_FIN':Dmax_25_50_mean_25_50_PN_fin}

MAX_MEAN_MAX = {'MAX_75_100_MEAN_75_100_TT_MAX':Dmax_75_100_mean_75_100_TT_max, 'MAX_75_100_MEAN_50_75_TT_MAX':Dmax_75_100_mean_50_75_TT_max, 'MAX_75_100_MEAN_25_50_TT_MAX':Dmax_75_100_mean_25_50_TT_max, 'MAX_50_75_MEAN_75_100_TT_MAX':Dmax_50_75_mean_75_100_TT_max, 'MAX_50_75_MEAN_50_75_TT_MAX':Dmax_50_75_mean_50_75_TT_max, 'MAX_50_75_MEAN_25_50_TT_MAX':Dmax_50_75_mean_25_50_TT_max, 'MAX_25_50_MEAN_75_100_TT_MAX':Dmax_25_50_mean_75_100_TT_max, 'MAX_25_50_MEAN_50_75_TT_MAX':Dmax_25_50_mean_50_75_TT_max, 'MAX_25_50_MEAN_25_50_TT_MAX':Dmax_25_50_mean_25_50_TT_max,
		   		'MAX_75_100_MEAN_75_100_PP_MAX':Dmax_75_100_mean_75_100_PP_max, 'MAX_75_100_MEAN_50_75_PP_MAX':Dmax_75_100_mean_50_75_PP_max, 'MAX_75_100_MEAN_25_50_PP_MAX':Dmax_75_100_mean_25_50_PP_max, 'MAX_50_75_MEAN_75_100_PP_MAX':Dmax_50_75_mean_75_100_PP_max, 'MAX_50_75_MEAN_50_75_PP_MAX':Dmax_50_75_mean_50_75_PP_max, 'MAX_50_75_MEAN_25_50_PP_MAX':Dmax_50_75_mean_25_50_PP_max, 'MAX_25_50_MEAN_75_100_PP_MAX':Dmax_25_50_mean_75_100_PP_max, 'MAX_25_50_MEAN_50_75_PP_MAX':Dmax_25_50_mean_50_75_PP_max, 'MAX_25_50_MEAN_25_50_PP_MAX':Dmax_25_50_mean_25_50_PP_max,
   		   		'MAX_75_100_MEAN_75_100_PN_MAX':Dmax_75_100_mean_75_100_PN_max, 'MAX_75_100_MEAN_50_75_PN_MAX':Dmax_75_100_mean_50_75_PN_max, 'MAX_75_100_MEAN_25_50_PN_MAX':Dmax_75_100_mean_25_50_PN_max, 'MAX_50_75_MEAN_75_100_PN_MAX':Dmax_50_75_mean_75_100_PN_max, 'MAX_50_75_MEAN_50_75_PN_MAX':Dmax_50_75_mean_50_75_PN_max, 'MAX_50_75_MEAN_25_50_PN_MAX':Dmax_50_75_mean_25_50_PN_max, 'MAX_25_50_MEAN_75_100_PN_MAX':Dmax_25_50_mean_75_100_PN_max, 'MAX_25_50_MEAN_50_75_PN_MAX':Dmax_25_50_mean_50_75_PN_max, 'MAX_25_50_MEAN_25_50_PN_MAX':Dmax_25_50_mean_25_50_PN_max}


MAX_DUR_INI = {'MAX_75_100_DUR_75_100_TT_INI':Dmax_75_100_lon_75_100_TT_ini, 'MAX_75_100_DUR_50_75_TT_INI':Dmax_75_100_lon_50_75_TT_ini, 'MAX_75_100_DUR_25_50_TT_INI':Dmax_75_100_lon_25_50_TT_ini, 'MAX_50_75_DUR_75_100_TT_INI':Dmax_50_75_lon_75_100_TT_ini, 'MAX_50_75_DUR_50_75_TT_INI':Dmax_50_75_lon_50_75_TT_ini, 'MAX_50_75_DUR_25_50_TT_INI':Dmax_50_75_lon_25_50_TT_ini, 'MAX_25_50_DUR_75_100_TT_INI':Dmax_25_50_lon_75_100_TT_ini, 'MAX_25_50_DUR_50_75_TT_INI':Dmax_25_50_lon_50_75_TT_ini, 'MAX_25_50_DUR_25_50_TT_INI':Dmax_25_50_lon_25_50_TT_ini,
		   	   'MAX_75_100_DUR_75_100_PP_INI':Dmax_75_100_lon_75_100_PP_ini, 'MAX_75_100_DUR_50_75_PP_INI':Dmax_75_100_lon_50_75_PP_ini, 'MAX_75_100_DUR_25_50_PP_INI':Dmax_75_100_lon_25_50_PP_ini, 'MAX_50_75_DUR_75_100_PP_INI':Dmax_50_75_lon_75_100_PP_ini, 'MAX_50_75_DUR_50_75_PP_INI':Dmax_50_75_lon_50_75_PP_ini, 'MAX_50_75_DUR_25_50_PP_INI':Dmax_50_75_lon_25_50_PP_ini, 'MAX_25_50_DUR_75_100_PP_INI':Dmax_25_50_lon_75_100_PP_ini, 'MAX_25_50_DUR_50_75_PP_INI':Dmax_25_50_lon_50_75_PP_ini, 'MAX_25_50_DUR_25_50_PP_INI':Dmax_25_50_lon_25_50_PP_ini,
   		   	   'MAX_75_100_DUR_75_100_PN_INI':Dmax_75_100_lon_75_100_PN_ini, 'MAX_75_100_DUR_50_75_PN_INI':Dmax_75_100_lon_50_75_PN_ini, 'MAX_75_100_DUR_25_50_PN_INI':Dmax_75_100_lon_25_50_PN_ini, 'MAX_50_75_DUR_75_100_PN_INI':Dmax_50_75_lon_75_100_PN_ini, 'MAX_50_75_DUR_50_75_PN_INI':Dmax_50_75_lon_50_75_PN_ini, 'MAX_50_75_DUR_25_50_PN_INI':Dmax_50_75_lon_25_50_PN_ini, 'MAX_25_50_DUR_75_100_PN_INI':Dmax_25_50_lon_75_100_PN_ini, 'MAX_25_50_DUR_50_75_PN_INI':Dmax_25_50_lon_50_75_PN_ini, 'MAX_25_50_DUR_25_50_PN_INI':Dmax_25_50_lon_25_50_PN_ini}

MAX_DUR_FIN = {'MAX_75_100_DUR_75_100_TT_FIN':Dmax_75_100_lon_75_100_TT_fin, 'MAX_75_100_DUR_50_75_TT_FIN':Dmax_75_100_lon_50_75_TT_fin, 'MAX_75_100_DUR_25_50_TT_FIN':Dmax_75_100_lon_25_50_TT_fin, 'MAX_50_75_DUR_75_100_TT_FIN':Dmax_50_75_lon_75_100_TT_fin, 'MAX_50_75_DUR_50_75_TT_FIN':Dmax_50_75_lon_50_75_TT_fin, 'MAX_50_75_DUR_25_50_TT_FIN':Dmax_50_75_lon_25_50_TT_fin, 'MAX_25_50_DUR_75_100_TT_FIN':Dmax_25_50_lon_75_100_TT_fin, 'MAX_25_50_DUR_50_75_TT_FIN':Dmax_25_50_lon_50_75_TT_fin, 'MAX_25_50_DUR_25_50_TT_FIN':Dmax_25_50_lon_25_50_TT_fin,
		   	   'MAX_75_100_DUR_75_100_PP_FIN':Dmax_75_100_lon_75_100_PP_fin, 'MAX_75_100_DUR_50_75_PP_FIN':Dmax_75_100_lon_50_75_PP_fin, 'MAX_75_100_DUR_25_50_PP_FIN':Dmax_75_100_lon_25_50_PP_fin, 'MAX_50_75_DUR_75_100_PP_FIN':Dmax_50_75_lon_75_100_PP_fin, 'MAX_50_75_DUR_50_75_PP_FIN':Dmax_50_75_lon_50_75_PP_fin, 'MAX_50_75_DUR_25_50_PP_FIN':Dmax_50_75_lon_25_50_PP_fin, 'MAX_25_50_DUR_75_100_PP_FIN':Dmax_25_50_lon_75_100_PP_fin, 'MAX_25_50_DUR_50_75_PP_FIN':Dmax_25_50_lon_50_75_PP_fin, 'MAX_25_50_DUR_25_50_PP_FIN':Dmax_25_50_lon_25_50_PP_fin,
   		   	   'MAX_75_100_DUR_75_100_PN_FIN':Dmax_75_100_lon_75_100_PN_fin, 'MAX_75_100_DUR_50_75_PN_FIN':Dmax_75_100_lon_50_75_PN_fin, 'MAX_75_100_DUR_25_50_PN_FIN':Dmax_75_100_lon_25_50_PN_fin, 'MAX_50_75_DUR_75_100_PN_FIN':Dmax_50_75_lon_75_100_PN_fin, 'MAX_50_75_DUR_50_75_PN_FIN':Dmax_50_75_lon_50_75_PN_fin, 'MAX_50_75_DUR_25_50_PN_FIN':Dmax_50_75_lon_25_50_PN_fin, 'MAX_25_50_DUR_75_100_PN_FIN':Dmax_25_50_lon_75_100_PN_fin, 'MAX_25_50_DUR_50_75_PN_FIN':Dmax_25_50_lon_50_75_PN_fin, 'MAX_25_50_DUR_25_50_PN_FIN':Dmax_25_50_lon_25_50_PN_fin}

MAX_DUR_MAX = {'MAX_75_100_DUR_75_100_TT_MAX':Dmax_75_100_lon_75_100_TT_max, 'MAX_75_100_DUR_50_75_TT_MAX':Dmax_75_100_lon_50_75_TT_max, 'MAX_75_100_DUR_25_50_TT_MAX':Dmax_75_100_lon_25_50_TT_max, 'MAX_50_75_DUR_75_100_TT_MAX':Dmax_50_75_lon_75_100_TT_max, 'MAX_50_75_DUR_50_75_TT_MAX':Dmax_50_75_lon_50_75_TT_max, 'MAX_50_75_DUR_25_50_TT_MAX':Dmax_50_75_lon_25_50_TT_max, 'MAX_25_50_DUR_75_100_TT_MAX':Dmax_25_50_lon_75_100_TT_max, 'MAX_25_50_DUR_50_75_TT_MAX':Dmax_25_50_lon_50_75_TT_max, 'MAX_25_50_DUR_25_50_TT_MAX':Dmax_25_50_lon_25_50_TT_max,
		   	   'MAX_75_100_DUR_75_100_PP_MAX':Dmax_75_100_lon_75_100_PP_max, 'MAX_75_100_DUR_50_75_PP_MAX':Dmax_75_100_lon_50_75_PP_max, 'MAX_75_100_DUR_25_50_PP_MAX':Dmax_75_100_lon_25_50_PP_max, 'MAX_50_75_DUR_75_100_PP_MAX':Dmax_50_75_lon_75_100_PP_max, 'MAX_50_75_DUR_50_75_PP_MAX':Dmax_50_75_lon_50_75_PP_max, 'MAX_50_75_DUR_25_50_PP_MAX':Dmax_50_75_lon_25_50_PP_max, 'MAX_25_50_DUR_75_100_PP_MAX':Dmax_25_50_lon_75_100_PP_max, 'MAX_25_50_DUR_50_75_PP_MAX':Dmax_25_50_lon_50_75_PP_max, 'MAX_25_50_DUR_25_50_PP_MAX':Dmax_25_50_lon_25_50_PP_max,
   		   	   'MAX_75_100_DUR_75_100_PN_MAX':Dmax_75_100_lon_75_100_PN_max, 'MAX_75_100_DUR_50_75_PN_MAX':Dmax_75_100_lon_50_75_PN_max, 'MAX_75_100_DUR_25_50_PN_MAX':Dmax_75_100_lon_25_50_PN_max, 'MAX_50_75_DUR_75_100_PN_MAX':Dmax_50_75_lon_75_100_PN_max, 'MAX_50_75_DUR_50_75_PN_MAX':Dmax_50_75_lon_50_75_PN_max, 'MAX_50_75_DUR_25_50_PN_MAX':Dmax_50_75_lon_25_50_PN_max, 'MAX_25_50_DUR_75_100_PN_MAX':Dmax_25_50_lon_75_100_PN_max, 'MAX_25_50_DUR_50_75_PN_MAX':Dmax_25_50_lon_50_75_PN_max, 'MAX_25_50_DUR_25_50_PN_MAX':Dmax_25_50_lon_25_50_PN_max}

"MEAN"

MEAN_DUR_INI = {'MEAN_75_100_DUR_75_100_TT_INI':Dmean_75_100_lon_75_100_TT_ini, 'MEAN_75_100_DUR_50_75_TT_INI':Dmean_75_100_lon_50_75_TT_ini, 'MEAN_75_100_DUR_25_50_TT_INI':Dmean_75_100_lon_25_50_TT_ini, 'MEAN_50_75_DUR_75_100_TT_INI':Dmean_50_75_lon_75_100_TT_ini, 'MEAN_50_75_DUR_50_75_TT_INI':Dmean_50_75_lon_50_75_TT_ini, 'MEAN_50_75_DUR_25_50_TT_INI':Dmean_50_75_lon_25_50_TT_ini, 'MEAN_25_50_DUR_75_100_TT_INI':Dmean_25_50_lon_75_100_TT_ini, 'MEAN_25_50_DUR_50_75_TT_INI':Dmean_25_50_lon_50_75_TT_ini, 'MEAN_25_50_DUR_25_50_TT_INI':Dmean_25_50_lon_25_50_TT_ini,
		   	    'MEAN_75_100_DUR_75_100_PP_INI':Dmean_75_100_lon_75_100_PP_ini, 'MEAN_75_100_DUR_50_75_PP_INI':Dmean_75_100_lon_50_75_PP_ini, 'MEAN_75_100_DUR_25_50_PP_INI':Dmean_75_100_lon_25_50_PP_ini, 'MEAN_50_75_DUR_75_100_PP_INI':Dmean_50_75_lon_75_100_PP_ini, 'MEAN_50_75_DUR_50_75_PP_INI':Dmean_50_75_lon_50_75_PP_ini, 'MEAN_50_75_DUR_25_50_PP_INI':Dmean_50_75_lon_25_50_PP_ini, 'MEAN_25_50_DUR_75_100_PP_INI':Dmean_25_50_lon_75_100_PP_ini, 'MEAN_25_50_DUR_50_75_PP_INI':Dmean_25_50_lon_50_75_PP_ini, 'MEAN_25_50_DUR_25_50_PP_INI':Dmean_25_50_lon_25_50_PP_ini,
   		   	    'MEAN_75_100_DUR_75_100_PN_INI':Dmean_75_100_lon_75_100_PN_ini, 'MEAN_75_100_DUR_50_75_PN_INI':Dmean_75_100_lon_50_75_PN_ini, 'MEAN_75_100_DUR_25_50_PN_INI':Dmean_75_100_lon_25_50_PN_ini, 'MEAN_50_75_DUR_75_100_PN_INI':Dmean_50_75_lon_75_100_PN_ini, 'MEAN_50_75_DUR_50_75_PN_INI':Dmean_50_75_lon_50_75_PN_ini, 'MEAN_50_75_DUR_25_50_PN_INI':Dmean_50_75_lon_25_50_PN_ini, 'MEAN_25_50_DUR_75_100_PN_INI':Dmean_25_50_lon_75_100_PN_ini, 'MEAN_25_50_DUR_50_75_PN_INI':Dmean_25_50_lon_50_75_PN_ini, 'MEAN_25_50_DUR_25_50_PN_INI':Dmean_25_50_lon_25_50_PN_ini}

MEAN_DUR_FIN = {'MEAN_75_100_DUR_75_100_TT_FIN':Dmean_75_100_lon_75_100_TT_fin, 'MEAN_75_100_DUR_50_75_TT_FIN':Dmean_75_100_lon_50_75_TT_fin, 'MEAN_75_100_DUR_25_50_TT_FIN':Dmean_75_100_lon_25_50_TT_fin, 'MEAN_50_75_DUR_75_100_TT_FIN':Dmean_50_75_lon_75_100_TT_fin, 'MEAN_50_75_DUR_50_75_TT_FIN':Dmean_50_75_lon_50_75_TT_fin, 'MEAN_50_75_DUR_25_50_TT_FIN':Dmean_50_75_lon_25_50_TT_fin, 'MEAN_25_50_DUR_75_100_TT_FIN':Dmean_25_50_lon_75_100_TT_fin, 'MEAN_25_50_DUR_50_75_TT_FIN':Dmean_25_50_lon_50_75_TT_fin, 'MEAN_25_50_DUR_25_50_TT_FIN':Dmean_25_50_lon_25_50_TT_fin,
		   	    'MEAN_75_100_DUR_75_100_PP_FIN':Dmean_75_100_lon_75_100_PP_fin, 'MEAN_75_100_DUR_50_75_PP_FIN':Dmean_75_100_lon_50_75_PP_fin, 'MEAN_75_100_DUR_25_50_PP_FIN':Dmean_75_100_lon_25_50_PP_fin, 'MEAN_50_75_DUR_75_100_PP_FIN':Dmean_50_75_lon_75_100_PP_fin, 'MEAN_50_75_DUR_50_75_PP_FIN':Dmean_50_75_lon_50_75_PP_fin, 'MEAN_50_75_DUR_25_50_PP_FIN':Dmean_50_75_lon_25_50_PP_fin, 'MEAN_25_50_DUR_75_100_PP_FIN':Dmean_25_50_lon_75_100_PP_fin, 'MEAN_25_50_DUR_50_75_PP_FIN':Dmean_25_50_lon_50_75_PP_fin, 'MEAN_25_50_DUR_25_50_PP_FIN':Dmean_25_50_lon_25_50_PP_fin,
   		   	    'MEAN_75_100_DUR_75_100_PN_FIN':Dmean_75_100_lon_75_100_PN_fin, 'MEAN_75_100_DUR_50_75_PN_FIN':Dmean_75_100_lon_50_75_PN_fin, 'MEAN_75_100_DUR_25_50_PN_FIN':Dmean_75_100_lon_25_50_PN_fin, 'MEAN_50_75_DUR_75_100_PN_FIN':Dmean_50_75_lon_75_100_PN_fin, 'MEAN_50_75_DUR_50_75_PN_FIN':Dmean_50_75_lon_50_75_PN_fin, 'MEAN_50_75_DUR_25_50_PN_FIN':Dmean_50_75_lon_25_50_PN_fin, 'MEAN_25_50_DUR_75_100_PN_FIN':Dmean_25_50_lon_75_100_PN_fin, 'MEAN_25_50_DUR_50_75_PN_FIN':Dmean_25_50_lon_50_75_PN_fin, 'MEAN_25_50_DUR_25_50_PN_FIN':Dmean_25_50_lon_25_50_PN_fin}

MEAN_DUR_MAX = {'MEAN_75_100_DUR_75_100_TT_MAX':Dmean_75_100_lon_75_100_TT_max, 'MEAN_75_100_DUR_50_75_TT_MAX':Dmean_75_100_lon_50_75_TT_max, 'MEAN_75_100_DUR_25_50_TT_MAX':Dmean_75_100_lon_25_50_TT_max, 'MEAN_50_75_DUR_75_100_TT_MAX':Dmean_50_75_lon_75_100_TT_max, 'MEAN_50_75_DUR_50_75_TT_MAX':Dmean_50_75_lon_50_75_TT_max, 'MEAN_50_75_DUR_25_50_TT_MAX':Dmean_50_75_lon_25_50_TT_max, 'MEAN_25_50_DUR_75_100_TT_MAX':Dmean_25_50_lon_75_100_TT_max, 'MEAN_25_50_DUR_50_75_TT_MAX':Dmean_25_50_lon_50_75_TT_max, 'MEAN_25_50_DUR_25_50_TT_MAX':Dmean_25_50_lon_25_50_TT_max,
		   	    'MEAN_75_100_DUR_75_100_PP_MAX':Dmean_75_100_lon_75_100_PP_max, 'MEAN_75_100_DUR_50_75_PP_MAX':Dmean_75_100_lon_50_75_PP_max, 'MEAN_75_100_DUR_25_50_PP_MAX':Dmean_75_100_lon_25_50_PP_max, 'MEAN_50_75_DUR_75_100_PP_MAX':Dmean_50_75_lon_75_100_PP_max, 'MEAN_50_75_DUR_50_75_PP_MAX':Dmean_50_75_lon_50_75_PP_max, 'MEAN_50_75_DUR_25_50_PP_MAX':Dmean_50_75_lon_25_50_PP_max, 'MEAN_25_50_DUR_75_100_PP_MAX':Dmean_25_50_lon_75_100_PP_max, 'MEAN_25_50_DUR_50_75_PP_MAX':Dmean_25_50_lon_50_75_PP_max, 'MEAN_25_50_DUR_25_50_PP_MAX':Dmean_25_50_lon_25_50_PP_max,
   		   	    'MEAN_75_100_DUR_75_100_PN_MAX':Dmean_75_100_lon_75_100_PN_max, 'MEAN_75_100_DUR_50_75_PN_MAX':Dmean_75_100_lon_50_75_PN_max, 'MEAN_75_100_DUR_25_50_PN_MAX':Dmean_75_100_lon_25_50_PN_max, 'MEAN_50_75_DUR_75_100_PN_MAX':Dmean_50_75_lon_75_100_PN_max, 'MEAN_50_75_DUR_50_75_PN_MAX':Dmean_50_75_lon_50_75_PN_max, 'MEAN_50_75_DUR_25_50_PN_MAX':Dmean_50_75_lon_25_50_PN_max, 'MEAN_25_50_DUR_75_100_PN_MAX':Dmean_25_50_lon_75_100_PN_max, 'MEAN_25_50_DUR_50_75_PN_MAX':Dmean_25_50_lon_50_75_PN_max, 'MEAN_25_50_DUR_25_50_PN_MAX':Dmean_25_50_lon_25_50_PN_max}


MEAN_MAX_INI = {'MEAN_75_100_MAX_75_100_TT_INI':Dmean_75_100_max_75_100_TT_ini, 'MEAN_75_100_MAX_50_75_TT_INI':Dmean_75_100_max_50_75_TT_ini, 'MEAN_75_100_MAX_25_50_TT_INI':Dmean_75_100_max_25_50_TT_ini, 'MEAN_50_75_MAX_75_100_TT_INI':Dmean_50_75_max_75_100_TT_ini, 'MEAN_50_75_MAX_50_75_TT_INI':Dmean_50_75_max_50_75_TT_ini, 'MEAN_50_75_MAX_25_50_TT_INI':Dmean_50_75_max_25_50_TT_ini, 'MEAN_25_50_MAX_75_100_TT_INI':Dmean_25_50_max_75_100_TT_ini, 'MEAN_25_50_MAX_50_75_TT_INI':Dmean_25_50_max_50_75_TT_ini, 'MEAN_25_50_MAX_25_50_TT_INI':Dmean_25_50_max_25_50_TT_ini,
		   	    'MEAN_75_100_MAX_75_100_PP_INI':Dmean_75_100_max_75_100_PP_ini, 'MEAN_75_100_MAX_50_75_PP_INI':Dmean_75_100_max_50_75_PP_ini, 'MEAN_75_100_MAX_25_50_PP_INI':Dmean_75_100_max_25_50_PP_ini, 'MEAN_50_75_MAX_75_100_PP_INI':Dmean_50_75_max_75_100_PP_ini, 'MEAN_50_75_MAX_50_75_PP_INI':Dmean_50_75_max_50_75_PP_ini, 'MEAN_50_75_MAX_25_50_PP_INI':Dmean_50_75_max_25_50_PP_ini, 'MEAN_25_50_MAX_75_100_PP_INI':Dmean_25_50_max_75_100_PP_ini, 'MEAN_25_50_MAX_50_75_PP_INI':Dmean_25_50_max_50_75_PP_ini, 'MEAN_25_50_MAX_25_50_PP_INI':Dmean_25_50_max_25_50_PP_ini,
   		   	    'MEAN_75_100_MAX_75_100_PN_INI':Dmean_75_100_max_75_100_PN_ini, 'MEAN_75_100_MAX_50_75_PN_INI':Dmean_75_100_max_50_75_PN_ini, 'MEAN_75_100_MAX_25_50_PN_INI':Dmean_75_100_max_25_50_PN_ini, 'MEAN_50_75_MAX_75_100_PN_INI':Dmean_50_75_max_75_100_PN_ini, 'MEAN_50_75_MAX_50_75_PN_INI':Dmean_50_75_max_50_75_PN_ini, 'MEAN_50_75_MAX_25_50_PN_INI':Dmean_50_75_max_25_50_PN_ini, 'MEAN_25_50_MAX_75_100_PN_INI':Dmean_25_50_max_75_100_PN_ini, 'MEAN_25_50_MAX_50_75_PN_INI':Dmean_25_50_max_50_75_PN_ini, 'MEAN_25_50_MAX_25_50_PN_INI':Dmean_25_50_max_25_50_PN_ini}

MEAN_MAX_FIN = {'MEAN_75_100_MAX_75_100_TT_FIN':Dmean_75_100_max_75_100_TT_fin, 'MEAN_75_100_MAX_50_75_TT_FIN':Dmean_75_100_max_50_75_TT_fin, 'MEAN_75_100_MAX_25_50_TT_FIN':Dmean_75_100_max_25_50_TT_fin, 'MEAN_50_75_MAX_75_100_TT_FIN':Dmean_50_75_max_75_100_TT_fin, 'MEAN_50_75_MAX_50_75_TT_FIN':Dmean_50_75_max_50_75_TT_fin, 'MEAN_50_75_MAX_25_50_TT_FIN':Dmean_50_75_max_25_50_TT_fin, 'MEAN_25_50_MAX_75_100_TT_FIN':Dmean_25_50_max_75_100_TT_fin, 'MEAN_25_50_MAX_50_75_TT_FIN':Dmean_25_50_max_50_75_TT_fin, 'MEAN_25_50_MAX_25_50_TT_FIN':Dmean_25_50_max_25_50_TT_fin,
		   	    'MEAN_75_100_MAX_75_100_PP_FIN':Dmean_75_100_max_75_100_PP_fin, 'MEAN_75_100_MAX_50_75_PP_FIN':Dmean_75_100_max_50_75_PP_fin, 'MEAN_75_100_MAX_25_50_PP_FIN':Dmean_75_100_max_25_50_PP_fin, 'MEAN_50_75_MAX_75_100_PP_FIN':Dmean_50_75_max_75_100_PP_fin, 'MEAN_50_75_MAX_50_75_PP_FIN':Dmean_50_75_max_50_75_PP_fin, 'MEAN_50_75_MAX_25_50_PP_FIN':Dmean_50_75_max_25_50_PP_fin, 'MEAN_25_50_MAX_75_100_PP_FIN':Dmean_25_50_max_75_100_PP_fin, 'MEAN_25_50_MAX_50_75_PP_FIN':Dmean_25_50_max_50_75_PP_fin, 'MEAN_25_50_MAX_25_50_PP_FIN':Dmean_25_50_max_25_50_PP_fin,
   		   	    'MEAN_75_100_MAX_75_100_PN_FIN':Dmean_75_100_max_75_100_PN_fin, 'MEAN_75_100_MAX_50_75_PN_FIN':Dmean_75_100_max_50_75_PN_fin, 'MEAN_75_100_MAX_25_50_PN_FIN':Dmean_75_100_max_25_50_PN_fin, 'MEAN_50_75_MAX_75_100_PN_FIN':Dmean_50_75_max_75_100_PN_fin, 'MEAN_50_75_MAX_50_75_PN_FIN':Dmean_50_75_max_50_75_PN_fin, 'MEAN_50_75_MAX_25_50_PN_FIN':Dmean_50_75_max_25_50_PN_fin, 'MEAN_25_50_MAX_75_100_PN_FIN':Dmean_25_50_max_75_100_PN_fin, 'MEAN_25_50_MAX_50_75_PN_FIN':Dmean_25_50_max_50_75_PN_fin, 'MEAN_25_50_MAX_25_50_PN_FIN':Dmean_25_50_max_25_50_PN_fin}

MEAN_MAX_MAX = {'MEAN_75_100_MAX_75_100_TT_MAX':Dmean_75_100_max_75_100_TT_max, 'MEAN_75_100_MAX_50_75_TT_MAX':Dmean_75_100_max_50_75_TT_max, 'MEAN_75_100_MAX_25_50_TT_MAX':Dmean_75_100_max_25_50_TT_max, 'MEAN_50_75_MAX_75_100_TT_MAX':Dmean_50_75_max_75_100_TT_max, 'MEAN_50_75_MAX_50_75_TT_MAX':Dmean_50_75_max_50_75_TT_max, 'MEAN_50_75_MAX_25_50_TT_MAX':Dmean_50_75_max_25_50_TT_max, 'MEAN_25_50_MAX_75_100_TT_MAX':Dmean_25_50_max_75_100_TT_max, 'MEAN_25_50_MAX_50_75_TT_MAX':Dmean_25_50_max_50_75_TT_max, 'MEAN_25_50_MAX_25_50_TT_MAX':Dmean_25_50_max_25_50_TT_max,
		   	    'MEAN_75_100_MAX_75_100_PP_MAX':Dmean_75_100_max_75_100_PP_max, 'MEAN_75_100_MAX_50_75_PP_MAX':Dmean_75_100_max_50_75_PP_max, 'MEAN_75_100_MAX_25_50_PP_MAX':Dmean_75_100_max_25_50_PP_max, 'MEAN_50_75_MAX_75_100_PP_MAX':Dmean_50_75_max_75_100_PP_max, 'MEAN_50_75_MAX_50_75_PP_MAX':Dmean_50_75_max_50_75_PP_max, 'MEAN_50_75_MAX_25_50_PP_MAX':Dmean_50_75_max_25_50_PP_max, 'MEAN_25_50_MAX_75_100_PP_MAX':Dmean_25_50_max_75_100_PP_max, 'MEAN_25_50_MAX_50_75_PP_MAX':Dmean_25_50_max_50_75_PP_max, 'MEAN_25_50_MAX_25_50_PP_MAX':Dmean_25_50_max_25_50_PP_max,
   		   	    'MEAN_75_100_MAX_75_100_PN_MAX':Dmean_75_100_max_75_100_PN_max, 'MEAN_75_100_MAX_50_75_PN_MAX':Dmean_75_100_max_50_75_PN_max, 'MEAN_75_100_MAX_25_50_PN_MAX':Dmean_75_100_max_25_50_PN_max, 'MEAN_50_75_MAX_75_100_PN_MAX':Dmean_50_75_max_75_100_PN_max, 'MEAN_50_75_MAX_50_75_PN_MAX':Dmean_50_75_max_50_75_PN_max, 'MEAN_50_75_MAX_25_50_PN_MAX':Dmean_50_75_max_25_50_PN_max, 'MEAN_25_50_MAX_75_100_PN_MAX':Dmean_25_50_max_75_100_PN_max, 'MEAN_25_50_MAX_50_75_PN_MAX':Dmean_25_50_max_50_75_PN_max, 'MEAN_25_50_MAX_25_50_PN_MAX':Dmean_25_50_max_25_50_PN_max}

"DUR"

DUR_MAX_INI = {'DUR_75_100_MAX_75_100_TT_INI':Dlon_75_100_max_75_100_TT_ini, 'DUR_75_100_MAX_50_75_TT_INI':Dlon_75_100_max_50_75_TT_ini, 'DUR_75_100_MAX_25_50_TT_INI':Dlon_75_100_max_25_50_TT_ini, 'DUR_50_75_MAX_75_100_TT_INI':Dlon_50_75_max_75_100_TT_ini, 'DUR_50_75_MAX_50_75_TT_INI':Dlon_50_75_max_50_75_TT_ini, 'DUR_50_75_MAX_25_50_TT_INI':Dlon_50_75_max_25_50_TT_ini, 'DUR_25_50_MAX_75_100_TT_INI':Dlon_25_50_max_75_100_TT_ini, 'DUR_25_50_MAX_50_75_TT_INI':Dlon_25_50_max_50_75_TT_ini, 'DUR_25_50_MAX_25_50_TT_INI':Dlon_25_50_max_25_50_TT_ini,
		   	   'DUR_75_100_MAX_75_100_PP_INI':Dlon_75_100_max_75_100_PP_ini, 'DUR_75_100_MAX_50_75_PP_INI':Dlon_75_100_max_50_75_PP_ini, 'DUR_75_100_MAX_25_50_PP_INI':Dlon_75_100_max_25_50_PP_ini, 'DUR_50_75_MAX_75_100_PP_INI':Dlon_50_75_max_75_100_PP_ini, 'DUR_50_75_MAX_50_75_PP_INI':Dlon_50_75_max_50_75_PP_ini, 'DUR_50_75_MAX_25_50_PP_INI':Dlon_50_75_max_25_50_PP_ini, 'DUR_25_50_MAX_75_100_PP_INI':Dlon_25_50_max_75_100_PP_ini, 'DUR_25_50_MAX_50_75_PP_INI':Dlon_25_50_max_50_75_PP_ini, 'DUR_25_50_MAX_25_50_PP_INI':Dlon_25_50_max_25_50_PP_ini,
   		   	   'DUR_75_100_MAX_75_100_PN_INI':Dlon_75_100_max_75_100_PN_ini, 'DUR_75_100_MAX_50_75_PN_INI':Dlon_75_100_max_50_75_PN_ini, 'DUR_75_100_MAX_25_50_PN_INI':Dlon_75_100_max_25_50_PN_ini, 'DUR_50_75_MAX_75_100_PN_INI':Dlon_50_75_max_75_100_PN_ini, 'DUR_50_75_MAX_50_75_PN_INI':Dlon_50_75_max_50_75_PN_ini, 'DUR_50_75_MAX_25_50_PN_INI':Dlon_50_75_max_25_50_PN_ini, 'DUR_25_50_MAX_75_100_PN_INI':Dlon_25_50_max_75_100_PN_ini, 'DUR_25_50_MAX_50_75_PN_INI':Dlon_25_50_max_50_75_PN_ini, 'DUR_25_50_MAX_25_50_PN_INI':Dlon_25_50_max_25_50_PN_ini}

DUR_MAX_FIN = {'DUR_75_100_MAX_75_100_TT_FIN':Dlon_75_100_max_75_100_TT_fin, 'DUR_75_100_MAX_50_75_TT_FIN':Dlon_75_100_max_50_75_TT_fin, 'DUR_75_100_MAX_25_50_TT_FIN':Dlon_75_100_max_25_50_TT_fin, 'DUR_50_75_MAX_75_100_TT_FIN':Dlon_50_75_max_75_100_TT_fin, 'DUR_50_75_MAX_50_75_TT_FIN':Dlon_50_75_max_50_75_TT_fin, 'DUR_50_75_MAX_25_50_TT_FIN':Dlon_50_75_max_25_50_TT_fin, 'DUR_25_50_MAX_75_100_TT_FIN':Dlon_25_50_max_75_100_TT_fin, 'DUR_25_50_MAX_50_75_TT_FIN':Dlon_25_50_max_50_75_TT_fin, 'DUR_25_50_MAX_25_50_TT_FIN':Dlon_25_50_max_25_50_TT_fin,
		   	   'DUR_75_100_MAX_75_100_PP_FIN':Dlon_75_100_max_75_100_PP_fin, 'DUR_75_100_MAX_50_75_PP_FIN':Dlon_75_100_max_50_75_PP_fin, 'DUR_75_100_MAX_25_50_PP_FIN':Dlon_75_100_max_25_50_PP_fin, 'DUR_50_75_MAX_75_100_PP_FIN':Dlon_50_75_max_75_100_PP_fin, 'DUR_50_75_MAX_50_75_PP_FIN':Dlon_50_75_max_50_75_PP_fin, 'DUR_50_75_MAX_25_50_PP_FIN':Dlon_50_75_max_25_50_PP_fin, 'DUR_25_50_MAX_75_100_PP_FIN':Dlon_25_50_max_75_100_PP_fin, 'DUR_25_50_MAX_50_75_PP_FIN':Dlon_25_50_max_50_75_PP_fin, 'DUR_25_50_MAX_25_50_PP_FIN':Dlon_25_50_max_25_50_PP_fin,
   		   	   'DUR_75_100_MAX_75_100_PN_FIN':Dlon_75_100_max_75_100_PN_fin, 'DUR_75_100_MAX_50_75_PN_FIN':Dlon_75_100_max_50_75_PN_fin, 'DUR_75_100_MAX_25_50_PN_FIN':Dlon_75_100_max_25_50_PN_fin, 'DUR_50_75_MAX_75_100_PN_FIN':Dlon_50_75_max_75_100_PN_fin, 'DUR_50_75_MAX_50_75_PN_FIN':Dlon_50_75_max_50_75_PN_fin, 'DUR_50_75_MAX_25_50_PN_FIN':Dlon_50_75_max_25_50_PN_fin, 'DUR_25_50_MAX_75_100_PN_FIN':Dlon_25_50_max_75_100_PN_fin, 'DUR_25_50_MAX_50_75_PN_FIN':Dlon_25_50_max_50_75_PN_fin, 'DUR_25_50_MAX_25_50_PN_FIN':Dlon_25_50_max_25_50_PN_fin}

DUR_MAX_MAX = {'DUR_75_100_MAX_75_100_TT_MAX':Dlon_75_100_max_75_100_TT_max, 'DUR_75_100_MAX_50_75_TT_MAX':Dlon_75_100_max_50_75_TT_max, 'DUR_75_100_MAX_25_50_TT_MAX':Dlon_75_100_max_25_50_TT_max, 'DUR_50_75_MAX_75_100_TT_MAX':Dlon_50_75_max_75_100_TT_max, 'DUR_50_75_MAX_50_75_TT_MAX':Dlon_50_75_max_50_75_TT_max, 'DUR_50_75_MAX_25_50_TT_MAX':Dlon_50_75_max_25_50_TT_max, 'DUR_25_50_MAX_75_100_TT_MAX':Dlon_25_50_max_75_100_TT_max, 'DUR_25_50_MAX_50_75_TT_MAX':Dlon_25_50_max_50_75_TT_max, 'DUR_25_50_MAX_25_50_TT_MAX':Dlon_25_50_max_25_50_TT_max,
		   	   'DUR_75_100_MAX_75_100_PP_MAX':Dlon_75_100_max_75_100_PP_max, 'DUR_75_100_MAX_50_75_PP_MAX':Dlon_75_100_max_50_75_PP_max, 'DUR_75_100_MAX_25_50_PP_MAX':Dlon_75_100_max_25_50_PP_max, 'DUR_50_75_MAX_75_100_PP_MAX':Dlon_50_75_max_75_100_PP_max, 'DUR_50_75_MAX_50_75_PP_MAX':Dlon_50_75_max_50_75_PP_max, 'DUR_50_75_MAX_25_50_PP_MAX':Dlon_50_75_max_25_50_PP_max, 'DUR_25_50_MAX_75_100_PP_MAX':Dlon_25_50_max_75_100_PP_max, 'DUR_25_50_MAX_50_75_PP_MAX':Dlon_25_50_max_50_75_PP_max, 'DUR_25_50_MAX_25_50_PP_MAX':Dlon_25_50_max_25_50_PP_max,
   		   	   'DUR_75_100_MAX_75_100_PN_MAX':Dlon_75_100_max_75_100_PN_max, 'DUR_75_100_MAX_50_75_PN_MAX':Dlon_75_100_max_50_75_PN_max, 'DUR_75_100_MAX_25_50_PN_MAX':Dlon_75_100_max_25_50_PN_max, 'DUR_50_75_MAX_75_100_PN_MAX':Dlon_50_75_max_75_100_PN_max, 'DUR_50_75_MAX_50_75_PN_MAX':Dlon_50_75_max_50_75_PN_max, 'DUR_50_75_MAX_25_50_PN_MAX':Dlon_50_75_max_25_50_PN_max, 'DUR_25_50_MAX_75_100_PN_MAX':Dlon_25_50_max_75_100_PN_max, 'DUR_25_50_MAX_50_75_PN_MAX':Dlon_25_50_max_50_75_PN_max, 'DUR_25_50_MAX_25_50_PN_MAX':Dlon_25_50_max_25_50_PN_max}


DUR_MEAN_INI = {'DUR_75_100_MEAN_75_100_TT_INI':Dlon_75_100_mean_75_100_TT_ini, 'DUR_75_100_MEAN_50_75_TT_INI':Dlon_75_100_mean_50_75_TT_ini, 'DUR_75_100_MEAN_25_50_TT_INI':Dlon_75_100_mean_25_50_TT_ini, 'DUR_50_75_MEAN_75_100_TT_INI':Dlon_50_75_mean_75_100_TT_ini, 'DUR_50_75_MEAN_50_75_TT_INI':Dlon_50_75_mean_50_75_TT_ini, 'DUR_50_75_MEAN_25_50_TT_INI':Dlon_50_75_mean_25_50_TT_ini, 'DUR_25_50_MEAN_75_100_TT_INI':Dlon_25_50_mean_75_100_TT_ini, 'DUR_25_50_MEAN_50_75_TT_INI':Dlon_25_50_mean_50_75_TT_ini, 'DUR_25_50_MEAN_25_50_TT_INI':Dlon_25_50_mean_25_50_TT_ini,
		   		'DUR_75_100_MEAN_75_100_PP_INI':Dlon_75_100_mean_75_100_PP_ini, 'DUR_75_100_MEAN_50_75_PP_INI':Dlon_75_100_mean_50_75_PP_ini, 'DUR_75_100_MEAN_25_50_PP_INI':Dlon_75_100_mean_25_50_PP_ini, 'DUR_50_75_MEAN_75_100_PP_INI':Dlon_50_75_mean_75_100_PP_ini, 'DUR_50_75_MEAN_50_75_PP_INI':Dlon_50_75_mean_50_75_PP_ini, 'DUR_50_75_MEAN_25_50_PP_INI':Dlon_50_75_mean_25_50_PP_ini, 'DUR_25_50_MEAN_75_100_PP_INI':Dlon_25_50_mean_75_100_PP_ini, 'DUR_25_50_MEAN_50_75_PP_INI':Dlon_25_50_mean_50_75_PP_ini, 'DUR_25_50_MEAN_25_50_PP_INI':Dlon_25_50_mean_25_50_PP_ini,
   		   		'DUR_75_100_MEAN_75_100_PN_INI':Dlon_75_100_mean_75_100_PN_ini, 'DUR_75_100_MEAN_50_75_PN_INI':Dlon_75_100_mean_50_75_PN_ini, 'DUR_75_100_MEAN_25_50_PN_INI':Dlon_75_100_mean_25_50_PN_ini, 'DUR_50_75_MEAN_75_100_PN_INI':Dlon_50_75_mean_75_100_PN_ini, 'DUR_50_75_MEAN_50_75_PN_INI':Dlon_50_75_mean_50_75_PN_ini, 'DUR_50_75_MEAN_25_50_PN_INI':Dlon_50_75_mean_25_50_PN_ini, 'DUR_25_50_MEAN_75_100_PN_INI':Dlon_25_50_mean_75_100_PN_ini, 'DUR_25_50_MEAN_50_75_PN_INI':Dlon_25_50_mean_50_75_PN_ini, 'DUR_25_50_MEAN_25_50_PN_INI':Dlon_25_50_mean_25_50_PN_ini}

DUR_MEAN_FIN = {'DUR_75_100_MEAN_75_100_TT_FIN':Dlon_75_100_mean_75_100_TT_fin, 'DUR_75_100_MEAN_50_75_TT_FIN':Dlon_75_100_mean_50_75_TT_fin, 'DUR_75_100_MEAN_25_50_TT_FIN':Dlon_75_100_mean_25_50_TT_fin, 'DUR_50_75_MEAN_75_100_TT_FIN':Dlon_50_75_mean_75_100_TT_fin, 'DUR_50_75_MEAN_50_75_TT_FIN':Dlon_50_75_mean_50_75_TT_fin, 'DUR_50_75_MEAN_25_50_TT_FIN':Dlon_50_75_mean_25_50_TT_fin, 'DUR_25_50_MEAN_75_100_TT_FIN':Dlon_25_50_mean_75_100_TT_fin, 'DUR_25_50_MEAN_50_75_TT_FIN':Dlon_25_50_mean_50_75_TT_fin, 'DUR_25_50_MEAN_25_50_TT_FIN':Dlon_25_50_mean_25_50_TT_fin,
		   		'DUR_75_100_MEAN_75_100_PP_FIN':Dlon_75_100_mean_75_100_PP_fin, 'DUR_75_100_MEAN_50_75_PP_FIN':Dlon_75_100_mean_50_75_PP_fin, 'DUR_75_100_MEAN_25_50_PP_FIN':Dlon_75_100_mean_25_50_PP_fin, 'DUR_50_75_MEAN_75_100_PP_FIN':Dlon_50_75_mean_75_100_PP_fin, 'DUR_50_75_MEAN_50_75_PP_FIN':Dlon_50_75_mean_50_75_PP_fin, 'DUR_50_75_MEAN_25_50_PP_FIN':Dlon_50_75_mean_25_50_PP_fin, 'DUR_25_50_MEAN_75_100_PP_FIN':Dlon_25_50_mean_75_100_PP_fin, 'DUR_25_50_MEAN_50_75_PP_FIN':Dlon_25_50_mean_50_75_PP_fin, 'DUR_25_50_MEAN_25_50_PP_FIN':Dlon_25_50_mean_25_50_PP_fin,
   		   		'DUR_75_100_MEAN_75_100_PN_FIN':Dlon_75_100_mean_75_100_PN_fin, 'DUR_75_100_MEAN_50_75_PN_FIN':Dlon_75_100_mean_50_75_PN_fin, 'DUR_75_100_MEAN_25_50_PN_FIN':Dlon_75_100_mean_25_50_PN_fin, 'DUR_50_75_MEAN_75_100_PN_FIN':Dlon_50_75_mean_75_100_PN_fin, 'DUR_50_75_MEAN_50_75_PN_FIN':Dlon_50_75_mean_50_75_PN_fin, 'DUR_50_75_MEAN_25_50_PN_FIN':Dlon_50_75_mean_25_50_PN_fin, 'DUR_25_50_MEAN_75_100_PN_FIN':Dlon_25_50_mean_75_100_PN_fin, 'DUR_25_50_MEAN_50_75_PN_FIN':Dlon_25_50_mean_50_75_PN_fin, 'DUR_25_50_MEAN_25_50_PN_FIN':Dlon_25_50_mean_25_50_PN_fin}

DUR_MEAN_MAX = {'DUR_75_100_MEAN_75_100_TT_MAX':Dlon_75_100_mean_75_100_TT_max, 'DUR_75_100_MEAN_50_75_TT_MAX':Dlon_75_100_mean_50_75_TT_max, 'DUR_75_100_MEAN_25_50_TT_MAX':Dlon_75_100_mean_25_50_TT_max, 'DUR_50_75_MEAN_75_100_TT_MAX':Dlon_50_75_mean_75_100_TT_max, 'DUR_50_75_MEAN_50_75_TT_MAX':Dlon_50_75_mean_50_75_TT_max, 'DUR_50_75_MEAN_25_50_TT_MAX':Dlon_50_75_mean_25_50_TT_max, 'DUR_25_50_MEAN_75_100_TT_MAX':Dlon_25_50_mean_75_100_TT_max, 'DUR_25_50_MEAN_50_75_TT_MAX':Dlon_25_50_mean_50_75_TT_max, 'DUR_25_50_MEAN_25_50_TT_MAX':Dlon_25_50_mean_25_50_TT_max,
		   		'DUR_75_100_MEAN_75_100_PP_MAX':Dlon_75_100_mean_75_100_PP_max, 'DUR_75_100_MEAN_50_75_PP_MAX':Dlon_75_100_mean_50_75_PP_max, 'DUR_75_100_MEAN_25_50_PP_MAX':Dlon_75_100_mean_25_50_PP_max, 'DUR_50_75_MEAN_75_100_PP_MAX':Dlon_50_75_mean_75_100_PP_max, 'DUR_50_75_MEAN_50_75_PP_MAX':Dlon_50_75_mean_50_75_PP_max, 'DUR_50_75_MEAN_25_50_PP_MAX':Dlon_50_75_mean_25_50_PP_max, 'DUR_25_50_MEAN_75_100_PP_MAX':Dlon_25_50_mean_75_100_PP_max, 'DUR_25_50_MEAN_50_75_PP_MAX':Dlon_25_50_mean_50_75_PP_max, 'DUR_25_50_MEAN_25_50_PP_MAX':Dlon_25_50_mean_25_50_PP_max,
   		   		'DUR_75_100_MEAN_75_100_PN_MAX':Dlon_75_100_mean_75_100_PN_max, 'DUR_75_100_MEAN_50_75_PN_MAX':Dlon_75_100_mean_50_75_PN_max, 'DUR_75_100_MEAN_25_50_PN_MAX':Dlon_75_100_mean_25_50_PN_max, 'DUR_50_75_MEAN_75_100_PN_MAX':Dlon_50_75_mean_75_100_PN_max, 'DUR_50_75_MEAN_50_75_PN_MAX':Dlon_50_75_mean_50_75_PN_max, 'DUR_50_75_MEAN_25_50_PN_MAX':Dlon_50_75_mean_25_50_PN_max, 'DUR_25_50_MEAN_75_100_PN_MAX':Dlon_25_50_mean_75_100_PN_max, 'DUR_25_50_MEAN_50_75_PN_MAX':Dlon_25_50_mean_50_75_PN_max, 'DUR_25_50_MEAN_25_50_PN_MAX':Dlon_25_50_mean_25_50_PN_max}


"Datos SST"

archivo = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/SST_DATA/sst_grid_025_fullworld_1998.nc')
lat       = archivo['latitude'][:]
lon       = archivo['longitude'][:]
lat_inf   = np.where(lat == 0)[0][0]
lat_sup   = np.where(lat == 25)[0][0]
lon_inf   = np.where(lon == -75)[0][0]
lon_sup   = np.where(lon == -110)[0][0]
lat       = lat[lat_sup:lat_inf+1]
lon       = lon[lon_sup:lon_inf+1]
dates_ERA = pd.date_range('1997-12-01 00:00:00', '2011-12-31 18:00:00', freq='6H')


"Seleccionando los datos"
for i in range(1997, 2012): # Se quiere obtener las fechas de ERA para saber cuandos slices debe tener la matríz que contendrá los datos

	archivo = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/SST_DATA/sst_grid_025_fullworld_'+str(i)+'.nc')

	if i == 1997:
		SST       = archivo.variables['sst'][:, lat_sup:lat_inf+1, lon_sup:lon_inf+1]
	else:
		SST       = np.concatenate((SST, archivo.variables['sst'][:, lat_sup:lat_inf+1, lon_sup:lon_inf+1]))

CICLO = ciclo_diurno_anual(SST, dates_ERA, len(lat), len(lon))

#==============================================================================
'Ploteando composites del geopotencial'
#==============================================================================

H_composites = ['-1.5d','-1.25d','-1d','-0.75d','-0.5d','-0.25d','0d','0.25d','0.5d','0.75d','1d','1.25d','1.5d', '1.75d', '2d', '2.25d', '2.5d', '2.75d', '3d', '3.25d', '3.5d', '3.75d', '4d', '4.25d', '4.5d', '4.75d', '5d', '5.25d', '5.5d', '5.75d', '6d', '6.25d', '6.5d', '6.75d', '7d', '7.25d', '7.5d', '7.75d', '8d']
MESES        = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

"Ploteando composites del geopotencial"
Dict_comp = {}
Dict_Num_fechas = {}
MIN_W = []
MAX_W = []
for ch in ['TT', 'PP', 'PN']:
    for cri1 in ['MEAN']:
    	for cri2 in ['DUR']:
			for perc1 in ['75_100', '50_75', '25_50']:
				for perc2 in ['75_100', '50_75', '25_50']:
					if cri1 != cri2:

						if cri1+'_'+cri2 == 'MAX_MEAN':
							FECHAS_INICIO = MAX_MEAN_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = MAX_MEAN_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = MAX_MEAN_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						elif cri1+'_'+cri2 == 'MAX_DUR':
							FECHAS_INICIO = MAX_DUR_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = MAX_DUR_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = MAX_DUR_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						elif cri1+'_'+cri2 == 'MEAN_MAX':
							FECHAS_INICIO = MEAN_MAX_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = MEAN_MAX_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = MEAN_MAX_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						elif cri1+'_'+cri2 == 'MEAN_DUR':
							FECHAS_INICIO = MEAN_DUR_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = MEAN_DUR_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = MEAN_DUR_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						elif cri1+'_'+cri2 == 'DUR_MAX':
							FECHAS_INICIO = DUR_MAX_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = DUR_MAX_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = DUR_MAX_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						elif cri1+'_'+cri2 == 'DUR_MEAN':
							FECHAS_INICIO = DUR_MEAN_INI[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_INI']
							FECHAS_FIN    = DUR_MEAN_FIN[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_FIN']
							FECHAS_MAX    = DUR_MEAN_MAX[cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ch+'_MAX']

						for st, ST in enumerate(MESES):
							fechas_max = [x for x in FECHAS_MAX if pd.Timestamp(x).month == st+1]

							if len(fechas_max) != 0:
								os.mkdir(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_SST_ESTACIONAL/'+ch+'/'+ST+'/'+ch+'_'+cri1+'_'+perc1+'_'+cri2+'_'+perc2, 0777)

								print ch+'_'+cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ST

								ANOM_SST   = np.zeros((len(H_composites),len(lat),len(lon)))

								for k, i in enumerate(range(-6,33)): # Todo este for es para meter los datos de cada evento
									anom_sst   = np.zeros((len(fechas_max), len(lat), len(lon)))
									for g, m in enumerate(fechas_max):

										date_evento = pd.Timestamp(m) + relativedelta(hours = 6*i)
										cuando      = np.where(dates_ERA == date_evento)[0][0]
										mes         = date_evento.month
										hora        = date_evento.hour

										clima       = CICLO[MESES[mes-1]+'_'+str(hora)]

										anom_sst[g,:,:]   = SST[cuando] - clima

									ANOM_SST[k]   = np.nanmean(anom_sst, axis=0)

								ANOM_SST[ANOM_SST < 273] = np.nan # Esto se hace porque al meter una matríz que continen valores enmascarados en otra, los valores enmascarados toman el valor de -32767
								ANOM_SST[ANOM_SST > 350] = np.nan

								Dict_comp.update({ch+'_'+cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ST:ANOM_SST})
								Dict_Num_fechas.update({ch+'_'+cri1+'_'+perc1+'_'+cri2+'_'+perc2+'_'+ST:len(fechas_max)})
								MIN_W.append(np.nanmin(ANOM_SST))
								MAX_W.append(np.nanmax(ANOM_SST))

min_W = np.nanmin(MIN_W)
max_W = np.nanmax(MAX_W)

for num, names in enumerate(Dict_comp.keys()):

	ANOM = Dict_comp[names]
	for k, i, h in zip(range(len(H_composites)), range(-6,33), H_composites):
		print names + ' ' + str(k) + ' ' + str(len(Dict_comp.keys())) + ' ' + str(num)
		Anom_sst = ANOM[k]
		Pos    = '%02d' % (k,)
		N_eventos = Dict_Num_fechas[names]
		titulo = u'Compuesto de anomalías de la SST - '+'(Max '+h+')'+ '\n' +names+' ('+str(N_eventos)+' eventos)'
		path   = u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/8_expo_2017/COMPOSITES_SST_ESTACIONAL/'+names[:2]+'/'+names[-3:]+'/'+names[:-4]+'/'+Pos+'_'+names[3:]+'_'+h
		plotear(lat[-1], lat[0], lon[0], lon[-1], 15, 15, lon, lat, Anom_sst, min_W, max_W, u'°C', titulo, path, C_T='k', wind=False, mapa_u=None, mapa_v=None)
