# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:34:25 2018

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
from mpl_toolkits.basemap import Basemap

"###########################################################    FUNCIONES   ##############################################################"

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

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
		CF1 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.viridis, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.viridis, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r
	else:
		CF1 = map.contourf(x,y,mapa, 20, cmap= plt.cm.rainbow, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, cmap= plt.cm.rainbow, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r

	cb1 = plt.colorbar(CF1, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cb1.set_label(unds)
	ax.set_title(titulo, size='15', color = C_T)

	if wind == True:
		Q = map.quiver(x[::2,::2], y[::2,::2], mapa_u[::2,::2], mapa_v[::2,::2], scale=20)
		plt.quiverkey(Q, 0.93, 0.05, 2, '2 m/s' )

	#map.fillcontinents(color='white')
	plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
	plt.close('all')


"#############################################    CICLO DIURNO DE CICLO ANUAL    ################################################"

"Se leen datos de presion a resolución de 0.25 grados"
# archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/PRESION-SEA-LEVEL-ERA/MSLP_025x025_0x40N_120_55W.nc')
# lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365


"Fechas"
# time    = archivo['time'][:]
# cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
# fechas  = [cdftime.num2date(x) for x in time]
# DATES   = pd.DatetimeIndex(fechas)


"Viento"
# prs = archivo.variables['msl'][:]/100. # Para pasarlo a hectopascales

"Se calcula ciclo anual de ciclo diurno"
# CICLO_PRESSURE = ciclo_diurno_anual(prs, DATES, len(lat), len(lon))


"Si no se tiene buen computador, léase dictionario con los ciclos que ya se han calculado anteriormente"

a = open('/home/yordan/UNAL/TESIS_MAESTRIA/19_expo_2018/ciclo_diurno_anual_sst_025_6h.bin', 'rb')

CICLO_SST = pickle.load(a)
a.close()

"################################################################################################################"

"Leyendo datos"
file    = nc.Dataset('.../SST_EOFs.nc')
lat     = file.variables['latitude'][:]
lon     = file.variables['longitude'][:] - 360
tempo   = file.variables['time'][:]
CDFtime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
DATES   = [CDFtime.num2date(x) for x in tempo]
DATES   = pd.DatetimeIndex(DATES)

"Fechas"
archivo = nc.Dataset('/home/yordan/UNAL/TESIS_MAESTRIA/DATOS/ERA_INTERIM/WIND/wind_big_area_925hPa.nc')
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
fechas  = pd.DatetimeIndex(fechas)

"Chorro y rezago"
ch = u'TT' #Selección de chorro

for l, rz in enumerate([0, 12, 24, 36, 48, 60, 72, 96, 120, 144, 168, 192, 216, 240, 288, 336, 480, 600, 720, 840, 960]):   #número de horas de rezago en horas, en multiplos de 6, empezando de cero

    "Fecha hasta donde se va a hacer HMM"
    
    pos_2017_12_31 = np.where(fechas == Timestamp('2017-12-31 18:00:00'))[0][0]
    FECHAS         = fechas[3 : pos_2017_12_31+1 : 4]# Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas

    "Lectura de Estados"
    rf     = open('/home/yordan/UNAL/TESIS_MAESTRIA/24_expo_2018/States_'+ch+'_anom_925.csv', 'r')

    reader = csv.reader(rf)
    states = [row for row in reader][1:]
    rf.close()

    states3 = np.array([int(x[2]) for x in states])

    "NM: Número de estados del modelo."
    NM = 3

    " Se arregla el vector de estados "

    if ch == 'TT' and NM == 3:
        states3[states3 == 2] = 33
        states3[states3 == 3] = 22

        states3[states3 == 22] = 2
        states3[states3 == 33] = 3

    elif ch == 'PP' and NM == 3:
        states3[states3 == 2] = 33
        states3[states3 == 3] = 22

        states3[states3 == 22] = 2
        states3[states3 == 33] = 3

    elif ch == 'PN' and NM == 3:
        states3[states3 == 1] = 22
        states3[states3 == 2] = 11

        states3[states3 == 11] = 1
        states3[states3 == 22] = 2


    Min_prs = []
    Max_prs = []

    Comp_sst = np.zeros((NM, len(lat), len(lon)))

    Ttl  = []
    path = []

    for k in range(1, NM+1):
        S  = k

        if NM == 3: # Se arregla el vector de estados
            ST = states3

        pos   = np.where(ST == S)[0]
        dates = FECHAS[pos]

        "Posiciones del nc donde se cumplen las fechas en el estado deseado"
        pos_nc = [np.where(DATES == d)[0][0] for d in dates]
        pos_nc = np.array(pos_nc) + rz//6

        "Se extraen datos para los compuestos"
        CompSST = np.zeros((len(pos_nc), len(lat), len(lon)))

        MESES        = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        for i, j in enumerate(pos_nc):
            mes = DATES[j].month
            sst = file.variables['sst'][j]

            cc_SST = CICLO_SST[MESES[mes-1]+'_18'] # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM

            CompSST[i] = sst - cc_SST

        CompSST[CompSST == -32767.0] = np.NAN
        "Se calcula las anomalías de la sst -> Se plotea"
        Comp_sst[k-1] = np.nanmean(CompSST, axis = 0);
        Min_prs.append(np.nanmin(Comp_sst[k-1])); Max_prs.append(np.nanmax(Comp_sst[k-1]))


        path.append('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/24_expo_2018/' + '%.2d' % l + '_' + ch + '_CompSST_JanDec_st'+str(S)+'_HMM'+str(NM)+'_rezago_'+str(rz)+'h_anom')

        if rz <= 72:
            Ttl.append('Jan-Dec SST Composite \n' + ch + ' - State ' + str(S) + ' (HMM ' + str(NM) + ') + ' + str(rz) + 'h')
        else:
            Ttl.append('Jan-Dec SST Composite \n' + ch + ' - State ' + str(S) + ' (HMM ' + str(NM) + ') + ' + str(rz/24) + 'd')

    if ch == 'TT':
        MIN = -0.42
        MAX = 0.15
    elif ch == 'PP':
        MIN = -0.18
        MAX = 0.26
    elif ch == 'PN':
        MIN = -0.18
        MAX = 0.22

    # Estado 1
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[0], np.min(Min_prs), np.max(Max_prs), 'C', Ttl[0], path[0], C_T='k')
    #plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[0], MIN, MAX, 'C', Ttl[0], path[0], C_T='k')

    # Estado 2
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[1], np.min(Min_prs), np.max(Max_prs), 'C', Ttl[1], path[1], C_T='k')
    #plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[1], MIN, MAX, 'C', Ttl[1], path[1], C_T='k')

    # Estado 3
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[2], np.min(Min_prs), np.max(Max_prs), 'C', Ttl[2], path[2], C_T='k')
    #plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_sst[2], MIN, MAX, 'C', Ttl[2], path[2], C_T='k')
