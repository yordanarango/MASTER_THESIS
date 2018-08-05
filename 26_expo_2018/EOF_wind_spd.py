# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 09:25:32 2016

@author: yordan
"""

from mpl_toolkits.basemap import Basemap
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import linalg as la
import pandas as pd
from netcdftime import utime
import matplotlib.colors as colors
import pickle


"##########################################   FUNCIONES   #########################################"

class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y))


"###################################################################################################"



"DATOS"
file = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area_925hPa.nc')

U   = file['u'][0]
Lat = file['latitude'][:]
Lon = file['longitude'][:] - 360

time    = file['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
dates   = [cdftime.num2date(x) for x in time]
dates   = pd.DatetimeIndex(dates) 

pos_lat_inf = np.where(Lat == 0)[0][0];    pos_lat_sup = np.where(Lat == 30)[0][0]
pos_lon_inf = np.where(Lon == -105)[0][0]; pos_lon_sup = np.where(Lon == -65)[0][0]

u   = U[pos_lat_sup : pos_lat_inf + 1, pos_lon_inf : pos_lon_sup + 1]
lat = Lat[pos_lat_sup : pos_lat_inf + 1]
lon = Lon[pos_lon_inf : pos_lon_sup + 1]



"MATRÍZ AUXILIAR"

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
wh = np.where(aux == 1)


"REMOVIENDO CICLO ANUAL DE LA VELOCIDAD DEL VIENTO"
print "Calculando ciclo"

def remov_ciclo_v(UyV, LAT, LON, wh, fechas):

	MAPA = np.zeros((len(fechas), len(LAT), len(LON)))
	DATA = pd.DataFrame(index=fechas, columns=['datos'])

	for l in range(len(wh[0])):
			u     = UyV.variables['u'][:len(fechas), wh[0][l], wh[1][l]] # Se mete dentro de la función, para sólo ingresar a las series de interés, de lo contrario habría que hacer una lectura de todos los pixeles, procedimiento que es muy pesado para el computador.
			v     = UyV.variables['v'][:len(fechas), wh[0][l], wh[1][l]]
			SERIE = np.sqrt(u*u+v*v)
			media = []

			ano = np.zeros(len(fechas))

			for i in range(1,13): # Se recorre cada mes
				pos = np.where(fechas.month == i)[0]
				media.append(np.nanmean(SERIE[pos])) # Se guarda el valor de la media
				ano[pos] = SERIE[pos] - np.nanmean(SERIE[pos])

			MAPA[:, wh[0][l], wh[1][l]] = ano

	return MAPA

SPD_ANO = remov_ciclo_v(file, lat, lon, wh, dates)

print "Resample de 6h a diario"

spd_ano = np.zeros((SPD_ANO.shape[0]/4, SPD_ANO.shape[1], SPD_ANO.shape[2]))

for i in range(SPD_ANO.shape[0]/4):
	spd_ano[i] = np.mean(SPD_ANO[i*4:i*4+4], axis=0)



"SELECCIÓN DE LAS SERIES"
def seleccion(wh, serie): #Selección de los datos que están en el dominio de interés de las EOF
	ar = []
	for i in range(len(wh[0])):
		ar.append(serie[:,wh[0][i],wh[1][i]])

	serie_seleccionada = np.array(ar)
	serie_seleccionada = serie_seleccionada.T

	return serie_seleccionada

spd_EOF = seleccion(wh, spd_ano) # Selección de datos de velocidad



"MATRIZ DE COVARIANZA"
print "calculando matriz de covarianza"
matrix_cov = np.dot(spd_EOF, spd_EOF.T) # Matríz de covarianza para los datos de velocidad



"EXTRACCIÓN DE VALORES Y VECTORES PROPIOS"
print "Calculando vectores y valores propios"
e_values, e_vect = np.linalg.eig(matrix_cov)

e_val = e_values.real; e_vec = e_vect.real



"SE GUARDAN VALORES Y VECTORES PROPIOS"
Dict = {'valores_propios':e_val, 'vectores_propios':e_vec}

punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/Dict_valores_vectores_propios.bin', 'wb')
pickle.dump(Dict, punto_bin)

punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/Dict_valores_vectores_propios.bin', 'wb')
pickle.dump(Dict, punto_bin)



"VARIANZA EXPLICADA"
sum_evals = np.sum(e_val.real)
var_exp = (e_val.real/sum_evals) * 100



"GRAFICA DE VARIANZA EXPLICADA"
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)
ax.plot(np.arange(1,11), var_exp[0:10], marker='o', color='r')
ax.set_xlabel('Component', size='14')
ax.set_ylabel('Variance [%]', size='14')
ax.grid(True)
ax.set_title('Explained Variance of the Wind', size='12')
plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/Varianza_explicada_WIND_925.png', dpi=100, bbox_inches='tight')
plt.show()



"CALCULO DE EOF's"
print "Calculando EOF's"
PC = np.dot(e_vec.T, spd_EOF) # Cálculo de PC's para datos de velocidad del viento



"VOLVIENDO A LA FORMA ORIGINAL DE LOS DATOS"
PrCo = np.zeros((spd_ano.shape[0], spd_ano.shape[1], spd_ano.shape[2]))
for i in range(len(wh[0])):
	PrCo[:, wh[0][i], wh[1][i]] = PC[:,i].real



"PLOTEA EOF"

for NC in range(4):

	box_lon = [-98.25, -101.75,  -80.25, -76.75, -98.25]
	box_lat = [17.75,   10.75,     0.0,  7.0,  17.75]

	fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
	ax = fig.add_axes([0.1,0.1,0.8,0.8])
	
	map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=30, llcrnrlon=-105, urcrnrlon=-65, resolution='i')
	lons,lats = np.meshgrid(lon,lat)
	x,y = map(lons,lats)
	
	bounds=np.linspace( np.min(PrCo[NC]) ,np.max(PrCo[NC]),20)
	bounds=np.around(bounds, decimals=2)
	csf=map.contourf(x,y, PrCo[NC], 20, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.RdYlBu_r,levels=bounds)
	cbar=plt.colorbar(csf, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cbar.set_label('Amplitude', fontsize='15')
	TT_lon,TT_lat = map(box_lon, box_lat)
	
	map.plot(TT_lon, TT_lat, marker=None, color='k')
	map.drawcoastlines(linewidth = 0.8)
	map.drawcountries(linewidth = 0.8)
	map.drawparallels(np.arange(0, 30, 10),labels=[1,0,0,1])
	map.drawmeridians(np.arange(-105,-65,10),labels=[1,0,0,1])
	
	ax.set_title('Wind EOF -'+str(NC+1)+' ['+'%.2f' % var_exp[NC]+'%]', size='15', weight='medium')
	plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/EOF'+str(NC+1)+'_Wind_925.png', dpi=100,bbox_inches='tight')
	plt.close('all')


