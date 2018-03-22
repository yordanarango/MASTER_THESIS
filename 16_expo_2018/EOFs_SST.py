# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:25:32 2016

@author: yordan
"""

#from mpl_toolkits.basemap import Basemap
#import matplotlib.pylab as pl
import numpy as np
import netCDF4 as nc
from netcdftime import utime
#import matplotlib.pyplot as plt
#from scipy import linalg as la
import pandas as pd
import pickle
#import xlrd
#from scipy.stats.stats import pearsonr
#import matplotlib.colors as colors

#class MidpointNormalize(colors.Normalize):
#    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
#        self.midpoint = midpoint
#        colors.Normalize.__init__(self, vmin, vmax, clip)

#    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
#        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
#        return np.ma.masked_array(np.interp(value, x, y))

#==============================================================================
'''LECTURA DE DATOS'''
#==============================================================================

archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/SST_EOFs.nc')
#SST     = archivo.variables['sst'][:]

Lat = archivo.variables["latitude"][:]
Lon = archivo.variables["longitude"][:]-360
time = archivo.variables["time"][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
dates   = pd.DatetimeIndex(fechas)

#=============================================================================
'''DOMINIO PARA EOF's'''
#==============================================================================

aux = np.zeros((len(Lat), len(Lon)))

a = np.where(Lon == -95.75)[0][0] # La posición de la coordenada longitudinal -95.75
b = np.where(Lat == 16.75) [0][0] # La posición de la coordenada latitudinal 16.75
lon_95_75 = np.where(Lon == -95.75)[0][0]
lon_77_75 = np.where(Lon == -75.75)[0][0]


d = 0
e = 0
for i, j in enumerate(Lon[lon_95_75:lon_77_75+1]):
    if j <= -79.5:
        if i%2 == 0:
            aux[b+i-(i//2):b+i-(i//2)+19,a+i] = 1
        else:
            aux[b+i-((i//2)+1):b+i-((i//2)+1)+19,a+i] = 1
    else:
        if i%2 == 0:
            aux[b+i-(i//2):b+i-(i//2)+17-(5*d),a+i] = 1
            d = d+1
        else:
            aux[b+i-((i//2)+1):b+i-((i//2)+1)+15-(5*e),a+i] = 1
            e = e+1

f = np.where(Lon == -97.5)[0][0] # La posición de la coordenada longitudinal -97.5
g = np.where(Lat == 13.25)[0][0] # La posición de la coordenada latitudinal 13.25
lon_97_5 = np.where(Lon == -97.5)[0][0]
lon_95_75 = np.where(Lon == -95.75)[0][0]


for k, l in enumerate(Lon[lon_97_5:lon_95_75+1]):
    if k%2 == 0:
        aux[g+k-(k//2)-(1+(5*(k//2))):g+k-(k//2)+1,f+k] = 1
    else:
        aux[g+k-(k//2)-((k//2)+(4*((k//2)+1))):g+k-(k//2)+1,f+k] = 1

prueba        = archivo.variables['sst'][0]
MA            = np.where(np.ma.getmask(prueba) == True)
aux[MA]       = np.NAN
aux[ aux==0 ] = np.NAN
wh            = np.where(aux == 1)

#==============================================================================
'''Removiendo Ciclo diurno y anual de la velocidad del viento'''
#==============================================================================

def remov_ciclo_v(UyV, LAT, LON, wh, fechas):
    MAPA = np.zeros((len(fechas), len(LAT), len(LON)))
    DATA = pd.DataFrame(index=fechas, columns=['datos'])
    MEDIA = np.zeros((len(wh[0]), 12, 4))
    for l in range(len(wh[0])):
            serie = UyV.variables['sst'][:, wh[0][l], wh[1][l]] # Se mete dentro de la función, para sólo ingresar a las series de interés, de lo contrario habría que hacer una lectura de todos los pixeles, procedimiento que es muy pesado para el computador.
            media = np.zeros((12,4)) # Las filas representan los meses, y las columnas los cuatro horarios dentro de un día

            for i in range(1,13): # Se recorre cada mes
                for j in range(0, 19, 6): # Se recorre cada horario en cada mes
                    pos    = np.where((fechas.month == i ) & (fechas.hour == j))[0]
                    media[i-1, j/6] = np.mean(serie[pos]) # Se guarda el valor de la media

            MEDIA[l] = media
            ano = np.zeros(len(fechas))
            DATA['datos'] = serie
            SERIE = DATA['datos']

            for n in range(len(ano)):
                ano[n] = SERIE[n]-media[SERIE.index[n].month-1, SERIE.index[n].hour/6]

            MAPA[:, wh[0][l], wh[1][l]] = ano

    return MAPA, MEDIA

sst_ano, sst_media = remov_ciclo_v(archivo, Lat, Lon, wh, dates)

#==============================================================================
'''SELECCIÓN DE LAS SERIES'''
#==============================================================================

wh = np.where(aux == 1)

def seleccion(wh, serie): #Selección de los datos que están en el dominio de interés de las EOF
    ar = []
    for i in range(len(wh[0])):
        ar.append(serie[:,wh[0][i],wh[1][i]])

    serie_seleccionada = np.array(ar)
    serie_seleccionada = serie_seleccionada.T

    return serie_seleccionada

sst_EOF = seleccion(wh, sst_ano) # Selección de datos de velocidad

#==============================================================================
'''MATRIZ DE COVARIANZA'''
#==============================================================================

matrix_cov = np.dot(sst_EOF, sst_EOF.T) # Matríz de covarianza para los datos de velocidad

#==============================================================================
'''Extracción de valores y vectores propios'''
#==============================================================================

e_values, e_vect = np.linalg.eig(matrix_cov)

e_val = e_values.real; e_vec = e_vect.real

#==============================================================================
'''Extracción de componentes principales'''
#==============================================================================

PC1 = e_vec.T[0]
PC2 = e_vec.T[1]
PC3 = e_vec.T[2]
PC4 = e_vec.T[3]

#==============================================================================
'''Varianza Explicada'''
#==============================================================================

sum_evals = np.sum(e_val.real)
var_exp   = (e_val.real/sum_evals) * 100

#==============================================================================
'''Gráfica de Varianza explicada'''
#==============================================================================

#fig = plt.figure(figsize=(8,6))
#ax = fig.add_subplot(111)
#ax.plot(np.arange(1,11), var_exp[0:10], marker='o', color='r')
#ax.set_xlabel('Component', size='12', style='oblique')
#ax.set_ylabel('Variance [%]', size='12', style='oblique')
#ax.grid(True)
#ax.set_title('SST Explained variance', size='12')
#plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/Varianza explicada'+'.png', dpi=100,bbox_inches='tight')
#plt.close('all')

#==============================================================================
'''Cálculo de EOF's'''
#==============================================================================

PC = np.dot(e_vec.T, spd_EOF) # Cálculo de PC's para datos de SST

#==============================================================================
'''Volviendo a la forma original de los datos'''
#==============================================================================

PrCo = np.zeros((sst_ano.shape[0], sst_ano.shape[1], sst_ano.shape[2]))
for i in range(len(wh[0])):
    PrCo[:, wh[0][i], wh[1][i]] = PC[:,i].real

#==============================================================================
'''PLOTEA EOF'''
#==============================================================================

#fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
#box_lon = [-97.5, -79.5, -77.75, -95.75, -97.5]
#box_lat = [13.25, 4.25, 7.75, 16.75, 13.25]
#map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=24, llcrnrlon=-105, urcrnrlon=-75, resolution='i')
#lons,lats = np.meshgrid(Lon,Lat)
#x,y = map(lons,lats)
#bounds = np.linspace( np.min(PrCo[2]) ,np.max(PrCo[2]),20)
#bounds = np.around(bounds, decimals=2)
#csf = map.contourf(x,y, PrCo[2], 20, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.RdYlBu_r,levels=bounds)
#cbar = plt.colorbar(csf,orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
#cbar.set_label('$Amplitud$', fontsize='15')
#TT_lon,TT_lat = map(box_lon, box_lat)
#map.plot(TT_lon, TT_lat, marker=None, color='k')
#map.drawcoastlines(linewidth = 0.8)
#map.drawcountries(linewidth = 0.8)
#map.drawparallels(np.arange(0, 24, 8),labels=[1,0,0,1])
#map.drawmeridians(np.arange(-105,-75,10),labels=[1,0,0,1])
#ax.set_title('EOF-3 [9.61%]', size='15', weight='medium')
#plt.show()

#===============================================================================
"Crea diccionario"
#===============================================================================

a    = np.array((1,2,3,4,5,6))
Dict = {'var_exp':var_exp, 'Pincipal_Comp':PrCo}

punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/Dict_PrCo.bin', 'wb')
pickle.dump(Dict, punto_bin)

punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/Dict_PrCo.bin', 'wb')
pickle.dump(Dict, punto_bin)
