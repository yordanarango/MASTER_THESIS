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
        Q = map.quiver(x[::4,::4], y[::4,::4], mapa_u[::4,::4], mapa_v[::4,::4], scale=20)
        plt.quiverkey(Q, 0.93, 0.05, 2, '2 m/s' )

    print "No copiar esta función para plotear mapas. Revisar la parte de los quiver"
    #map.fillcontinents(color='white')
    plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
    plt.close('all')


"#############################################    CICLO DIURNO DE CICLO ANUAL    ################################################"

"Se leen datos de viento a resolución de 0.25 grados"
# archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')
# lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365


"Fechas"
# time    = archivo['time'][:]
# cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
# fechas  = [cdftime.num2date(x) for x in time]
# DATES   = pd.DatetimeIndex(fechas)[:]


"Viento"
# v   = archivo['v10'][:]
# u   = archivo['u10'][:]
# wnd = np.sqrt(v*v+u*u)


"Se calcula ciclo anual de ciclo diurno"
# CICLO_WIND = ciclo_diurno_anual(wnd, DATES, len(lat), len(lon))


"Si no se tiene buen computador, léase dictionario con los ciclos que ya se han calculado anteriormente"

a = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/22_expo_2018/ciclo_diurno_anual_wind_025_6h_big_area.bin', 'rb')
b = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/22_expo_2018/ciclo_diurno_anual_U_025_6h_big_area.bin', 'rb')
c = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/22_expo_2018/ciclo_diurno_anual_V_025_6h_big_area.bin', 'rb')

CICLO_WIND = pickle.load(a)
CICLO_U    = pickle.load(b)
CICLO_V    = pickle.load(c)

"################################################################################################################"

"Chorro Y Barra de colores"
ch = 'PN' #Selección de chorro
BC = 'individual' # individual o conjunto

"Leyendo datos"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area.nc')
lat = archivo.variables['latitude'][:]       # va desde 7°N hasta 25°N
lon = archivo.variables['longitude'][:]-360  # va desde -64.5°W hasta -101°W

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[:] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 18:00 horas

"Selección de fechas en Noviembre, Diciembre, Enero, Febrero, Marzo, Abril"
FCH = DATES[3::4]
DT  = []

leap_years   = np.array([x for x in set(FCH[FCH.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(FCH[~FCH.is_leap_year].year)])  # Años normales
years        = np.array([x for x in set(FCH.year)])

for i in years[:-1]:

    pos = np.where(FCH == pd.Timestamp(str(i)+'-11-01 18:00:00'))[0][0]

    if np.any(normal_years == i+1) == True:
    	DT.append(FCH[pos:pos+181])
    else:
    	DT.append(FCH[pos:pos+182])

FECHAS = pd.DatetimeIndex(np.concatenate(DT))


"Lectura de Estados"
rf     = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/23_expo_2018/States_'+ch+'_NovAbr_anom.csv', 'r')

reader = csv.reader(rf)
states = [row for row in reader][1:]
rf.close()

states3 = np.array([int(x[2]) for x in states])


"Se extraen fechas de interés: NM: Número de estados del modelo. S: estado deseado"
NM = 3

" Se arregla el vector de estados "
print "En el estado tres no hay que arreglar nada en ninguno de los chorros"

Min_spd = []
Max_spd = []

Comp_spd = np.zeros((NM, len(lat), len(lon)))
Comp_u   = np.zeros((NM, len(lat), len(lon)))
Comp_v   = np.zeros((NM, len(lat), len(lon)))

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

    "Se extraen datos para los compuestos"
    CompU = np.zeros((len(pos_nc), len(lat), len(lon)))
    CompV = np.zeros((len(pos_nc), len(lat), len(lon)))

    MESES        = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    for i, j in enumerate(pos_nc):
        mes = DATES[j].month
        u   = archivo.variables['u10'][j]
        v   = archivo.variables['v10'][j]

        cc_U = CICLO_U[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
    	cc_V = CICLO_V[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM

    	CompU[i] = u - cc_U
    	CompV[i] = v - cc_V

    "Se calcula la velocidad de las anomalías del viento. -> Se plotea"
    Comp_u[k-1] = np.mean(CompU, axis = 0); Comp_v[k-1] = np.mean(CompV, axis = 0);
    Comp_spd[k-1] = np.sqrt(Comp_u[k-1] * Comp_u[k-1] + Comp_v[k-1] * Comp_v[k-1]);
    Min_spd.append(np.min(Comp_spd[k-1])); Max_spd.append(np.max(Comp_spd[k-1]));

    path.append('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/23_expo_2018/WND_Composites/' + ch + '_CompWind_NovAbr_st'+str(S)+'_HMM'+str(NM)+'_'+BC+'_anom')
    Ttl.append('Nov-Abr Wind Anomalies Composite \n' + ch + ' - State ' + str(S) + ' (HMM ' + str(NM) + ')')

if BC == 'conjunto':
    # Estado 1
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[0], np.min(Min_spd), np.max(Max_spd), 'm/s', Ttl[0], path[0], C_T='k', wind=True, mapa_u=Comp_u[0], mapa_v=Comp_v[0])
    # Estado 2
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[1], np.min(Min_spd), np.max(Max_spd), 'm/s', Ttl[1], path[1], C_T='k', wind=True, mapa_u=Comp_u[1], mapa_v=Comp_v[1])
    # Estado 3
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[2], np.min(Min_spd), np.max(Max_spd), 'm/s', Ttl[2], path[2], C_T='k', wind=True, mapa_u=Comp_u[2], mapa_v=Comp_v[2])
elif BC == 'individual':
    # Estado 1
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[0], Min_spd[0], Max_spd[0], 'm/s', Ttl[0], path[0], C_T='k', wind=True, mapa_u=Comp_u[0], mapa_v=Comp_v[0])
    # Estado 2
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[1], Min_spd[1], Max_spd[1], 'm/s', Ttl[1], path[1], C_T='k', wind=True, mapa_u=Comp_u[1], mapa_v=Comp_v[1])
    # Estado 3
    plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon, lat, Comp_spd[2], Min_spd[2], Max_spd[2], 'm/s', Ttl[2], path[2], C_T='k', wind=True, mapa_u=Comp_u[2], mapa_v=Comp_v[2])
