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

def plotear(lllat, urlat, lllon, urlon, dist_lat, dist_lon, Lon, Lat, mapa, bar_min, bar_max, unds, titulo, path, C_T='k', wind=False, p=False, mapa_u=None, mapa_v=None, p_x=None, p_y=None):

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
	# p                      : boolean que diga si se va a pintar un poligono (u punto tambien es permitido) 
	# mapa_u                 : array de numpay 2D con la componente en x de la velocidad y que será utilizado para pintar las flechas. Este tomara algun valor siempre y cuando wind=True
	# mapa_v                 : array de numpay 2D con la componente en y de la velocidad y que será utilizado para pintar las flechas. Este tomara algun valor siempre y cuando wind=True
	# p_x                    : array o float que diga en que longitud se va a pintar el poligono (array) o el punto (float)
	# p_y                    : array o float que diga en que latitud se va a pintar el poligono (array) o el punto (float)

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
		CF1 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.rainbow, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.rainbow, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r
	else:
		CF1 = map.contourf(x,y,mapa, 20, cmap= plt.cm.rainbow, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, cmap= plt.cm.rainbow, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r

	cb1 = plt.colorbar(CF1, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cb1.set_label(unds)
	ax.set_title(titulo, size='15', color = C_T)

	if wind == True:
		Q = map.quiver(x[::2,::2], y[::2,::2], mapa_u[::2,::2], mapa_v[::2,::2], scale=20)
		plt.quiverkey(Q, 0.93, 0.05, 2, '2 m/s' )

	if p == True:
		p_lon, p_lat = map(p_x, p_y)

		map.plot(p_lon, p_lat, marker=None, color='k')

	#map.fillcontinents(color='white')
	plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
	plt.close('all')

"#############################################    DATOS    ################################################"

file = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area_925hPa.nc')

U   = file['u'][0]
Lat = file['latitude'][:]
Lon = file['longitude'][:] - 360

u   = U[10:, 60:210]
lat = Lat[10:]
lon = Lon[60:210]

path = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/prueba'
P_X  = [-98.25, -101.75,  -80.25, -76.75, -98.25] 
P_Y  = [17.75,   10.75,     0.0,  7.0,  17.75]
plotear(lat[-1], lat[0], lon[0], lon[-1], 5, 5, lon, lat, u, np.nanmin(u), np.nanmax(u), 'm/s', 'dominio', path, p=True, p_x=P_X, p_y=P_Y)


"#############################################    MATRIZ AUXILIAR    ################################################"

aux       = np.zeros((len(lat), len(lon)))


"Parte central de la matriz aux"

lon_ini_1 = np.where(lon == -98.25)[0][0]
lon_fin_1 = np.where(lon == -80.25)[0][0]

lat_corn_1  = 17.75

for i, j in enumerate(lon[lon_ini_1 : lon_fin_1 + 1]):

	lat_corn_1  = lat_corn_1 - 0.25 * (i % 2)
	
	pos_lon_1 = np.where(lon == j)[0][0]
	pos_lat_1 = np.where(lat == lat_corn_1)[0][0]
	
	aux[pos_lat_1 : pos_lat_1 + 36, pos_lon_1] = 1



"Parte izquierda de aux"
lat_corn_2 = 9

for i, j in enumerate(np.linspace(-98.5, -101.75, 14)):

	lat_corn_2 = lat_corn_2 + 0.25 * (i % 2)
	
	pos_lon_2 = np.where(lon == j)[0][0]
	pos_lat_2 = np.where(lat == lat_corn_2)[0][0]

	aux[pos_lat_2 - 37 + ((i + 1) // 2) + 2 * (i + 1) : pos_lat_2 + 1, pos_lon_2] = 1



"Parte derechaa de aux"
lat_corn_3 = 8.5

for i, j in enumerate(np.linspace(-80.0, -76.75, 14)):

	pos_lon_3 = np.where(lon == j)
	pos_lat_3 = np.where(lat == lat_corn_3)[0][0]

	aux[pos_lat_3 : pos_lat_3 + 37 - (i // 2) - 2 * (i + 1), pos_lon_3] = 1

	lat_corn_3 = lat_corn_3 - 0.25 * (i % 2)


aux[aux == 0] = np.nan
u_aux = u * aux

path = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/prueba_aux'
P_X  = [-98.25, -101.75,  -80.25, -76.75, -98.25] 
P_Y  = [17.75,   10.75,     0.0,  7.0,  17.75]
plotear(lat[-1], lat[0], lon[0], lon[-1], 5, 5, lon, lat, u_aux, np.nanmin(u), np.nanmax(u), 'm/s', 'dominio aux', path, p=False, p_x=P_X, p_y=P_Y)

