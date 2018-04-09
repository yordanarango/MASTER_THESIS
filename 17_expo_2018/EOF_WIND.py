# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:25:32 2016

@author: yordan
"""

from mpl_toolkits.basemap import Basemap
#import matplotlib.pylab as pl
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
#from scipy import linalg as la
import pandas as pd
import pickle
from netcdftime import utime
import matplotlib.colors as colors

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#==============================================================================
'''LECTURA DE DATOS'''
#==============================================================================

UyV = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/CORRELACIONES_SST-SSP_vsCPs/UyV-025x025-6h.nc')
Variables = [v for v in UyV.variables]

Lat = UyV.variables["latitude"][:]
Lon = UyV.variables["longitude"][:]-360
time = UyV.variables["time"][:]
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

aux[ aux==0 ] = np.NAN
wh = np.where(aux == 1)

#==============================================================================
'''Removiendo anual de la velocidad del viento'''
#==============================================================================

print "Calculando ciclo"

def remov_ciclo_v(UyV, LAT, LON, wh, fechas):
    MAPA = np.zeros((len(fechas), len(LAT), len(LON)))
    DATA = pd.DataFrame(index=fechas, columns=['datos'])
    MEDIA = np.zeros((len(wh[0]), 12))
    for l in range(len(wh[0])):
            u = UyV.variables['u10'][:len(fechas), wh[0][l], wh[1][l]] # Se mete dentro de la función, para sólo ingresar a las series de interés, de lo contrario habría que hacer una lectura de todos los pixeles, procedimiento que es muy pesado para el computador.
            v = UyV.variables['v10'][:len(fechas), wh[0][l], wh[1][l]]
            serie = np.sqrt(u*u+v*v)
            media = []

            for i in range(1,13): # Se recorre cada mes
                pos = np.where(fechas.month == i)[0]
                media.append(np.mean(serie[pos])) # Se guarda el valor de la media

            media    = np.array(media)
            MEDIA[l] = media
            ano = np.zeros(len(fechas))
            DATA['datos'] = serie
            SERIE = DATA['datos'][fechas[0]:fechas[-1]]
            for n in range(len(ano)):
                ano[n] = SERIE[n]-media[SERIE.index[n].month-1]

            MAPA[:, wh[0][l], wh[1][l]] = ano

    return MAPA, MEDIA

SPD_ANO, spd_media = remov_ciclo_v(UyV, Lat, Lon, wh, dates)

print "Resample de 6h a diario"
spd_ano = np.zeros((SPD_ANO.shape[0]/4, SPD_ANO.shape[1], SPD_ANO.shape[2]))
for i in range(SPD_ANO.shape[0]/4):
	spd_ano[i] = np.mean(SPD_ANO[i*4:i*4+4], axis=0)

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

spd_EOF = seleccion(wh, spd_ano) # Selección de datos de velocidad
#meridional_EOF = seleccion(wh, meridional_ano) # Selección de datos de componente meridional de la velocidad

#==============================================================================
'''MATRIZ DE COVARIANZA'''
#==============================================================================

print "calculando matriz de covarianza"
matrix_cov = np.dot(spd_EOF, spd_EOF.T) # Matríz de covarianza para los datos de velocidad
#matrix_cov = np.dot(meridional_EOF, meridional_EOF.T) # Matríz de covarianza para los datos de componente meridional del viento

#==============================================================================
'''Extracción de valores y vectores propios'''
#==============================================================================

print "Calculando vectores y valores propios"
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
var_exp = (e_val.real/sum_evals) * 100

#==============================================================================
'''Gráfica de Varianza explicada'''
#==============================================================================

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.plot(np.arange(1,11), var_exp[0:10], marker='o', color='r')
ax.set_xlabel('Component', size='12', style='oblique')
ax.set_ylabel('Variance [%]', size='12', style='oblique')
ax.grid(True)
ax.set_title('Explained Wind Variance', size='12')
plt.savefig('/home/yordan/Escritorio/Varianza_explicada_WIND.png', dpi=100,bbox_inches='tight')
plt.show()

#==============================================================================
'''Cálculo de EOF's'''
#==============================================================================
print "Calculando EOF's"
PC = np.dot(e_vec.T, spd_EOF) # Cálculo de PC's para datos de velocidad del viento
#PC = np.dot(e_vec.T, meridional_EOF) # Cálculo de PC's para datos de componente meridional del viento

#==============================================================================
'''Volviendo a la forma original de los datos'''
#==============================================================================

# Se utiliza cualquiera uno de los dos PrCo, según la serie que se esté utilizando: meridional_ano ó spd_ano
PrCo = np.zeros((spd_ano.shape[0], spd_ano.shape[1], spd_ano.shape[2]))
#PrCo = np.zeros((meridional_ano.shape[0], meridional_ano.shape[1], meridional_ano.shape[2]))
for i in range(len(wh[0])):
    PrCo[:, wh[0][i], wh[1][i]] = PC[:,i].real


#==============================================================================
'''PLOTEA EOF'''
#==============================================================================

NC = 3

fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
ax = fig.add_axes([0.1,0.1,0.8,0.8])
box_lon = [-97.5, -79.5, -77.75, -95.75, -97.5]
box_lat = [13.25, 4.25, 7.75, 16.75, 13.25]
map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=24, llcrnrlon=-105, urcrnrlon=-75, resolution='i')
lons,lats = np.meshgrid(Lon,Lat)
x,y = map(lons,lats)
bounds=np.linspace( np.min(PrCo[NC]) ,np.max(PrCo[NC]),20)
bounds=np.around(bounds, decimals=2)
csf=map.contourf(x,y, PrCo[NC], 20, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.RdYlBu_r,levels=bounds)
cbar=plt.colorbar(csf,orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
cbar.set_label('$Amplitud$', fontsize='15')
TT_lon,TT_lat = map(box_lon, box_lat)
map.plot(TT_lon, TT_lat, marker=None, color='k')
map.drawcoastlines(linewidth = 0.8)
map.drawcountries(linewidth = 0.8)
map.drawparallels(np.arange(0, 24, 8),labels=[1,0,0,1])
map.drawmeridians(np.arange(-105,-75,10),labels=[1,0,0,1])
ax.set_title('Wind EOF-'+str(NC+1)+' ['+'%.2f' % var_exp[NC]+'%]', size='15', weight='medium')
plt.savefig('/home/yordan/Escritorio/EOF'+str(NC+1)+'_Wind.png', dpi=100,bbox_inches='tight')
#plt.show()

#==============================================================================
'''MAPA AREA EOF's'''
#==============================================================================

#area = UyV.variables['v10'][0]*aux

#box_lon = [-97.5, -79.5, -77.75, -95.75, -97.5]
#box_lat = [13.25, 4.25, 7.75, 16.75, 13.25]
#lons,lats = np.meshgrid(Lon,Lat)
#fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
#ax = fig.add_axes([0.1,0.1,0.8,0.8])
#map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=20, llcrnrlon=-100, urcrnrlon=-75, resolution='i')
#map.drawcoastlines(linewidth = 0.8)
#map.drawcountries(linewidth = 0.8)
#map.drawparallels(np.arange(0, 20, 7),labels=[1,0,0,1])
#map.drawmeridians(np.arange(-100,-75,10),labels=[1,0,0,1])
#x,y = map(lons,lats)
#TT_lon,TT_lat = map(box_lon, box_lat)
#CF = map.contourf(x,y, area[:,:], np.linspace(-7, 5, 20), extend='both', cmap=plt.cm.RdYlBu_r)#plt.cm.rainbow
#cb = map.colorbar(CF, size="5%", pad="2%", extendrect = 'True', drawedges = 'True', format='%.1f')
#cb.set_label('m/s')
#ax.set_title('$Meridional$ $Speed$ $(1979-01-01)$', size='15', weight='medium')
#map.plot(TT_lon, TT_lat, marker=None, color='k')
#map.fillcontinents(color='white')
#plt.show()

print "Se escriben diccionarios"

Dict = {'var_exp':var_exp, 'Pincipal_Comp':PrCo}

punto_bin = open('/home/yordan/Escritorio/Dict_PrCo_wind.bin', 'wb')
pickle.dump(PrCo, punto_bin)

punto_bin = open('/home/yordan/Escritorio/Dict_PrCo_wind.bin', 'wb')
pickle.dump(PrCo, punto_bin)
