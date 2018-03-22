# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 13:34:25 2018

@author: yordan
"""
from hmmlearn.hmm import GaussianHMM
import numpy as np
import netCDF4 as nc
from netcdftime import utime
from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


"FUNCIONES"
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
		CF1 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.RdYlBu_r, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, norm=MidpointNormalize(midpoint=0), cmap= plt.cm.RdYlBu_r, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r
	else:
		CF1 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='max')#plt.cm.rainbow , plt.cm.RdYlBu_r
		CF2 = map.contourf(x,y,mapa, 20, cmap= plt.cm.jet, levels=bounds, extend='min')#plt.cm.rainbow, plt.cm.RdYlBu_r

	cb1 = plt.colorbar(CF1, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
	cb1.set_label(unds)
	cb1.ax.tick_params(labelsize=9) 
	ax.set_title(titulo, size='15', color = C_T)

	if wind == True:
		Q = map.quiver(x[::2,::2], y[::2,::2], mapa_u[::2,::2], mapa_v[::2,::2], scale=150)
		plt.quiverkey(Q, 0.93, 0.05, 10, '10 m/s' )

	#map.fillcontinents(color='white')
	plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
	plt.close('all')

"###################################################################################################################################"

archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-360


"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[3::4] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas

"Viento"
pos_lon = np.where(lon == -79.5)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == 7)[0][0]
v   = archivo['v10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas
u   = archivo['u10'][3::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad, la cual corresponde a las 16:00 horas
wnd = np.sqrt(v*v+u*u)

"Se seleccionan años NO bisiestos"
DT  = [] 
WND = []
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales. NO bisiestos

for i, d in enumerate(DATES):
	if np.any(normal_years == d.year):
		DT.append(d)
		WND.append(wnd[i])

dates = pd.DatetimeIndex(DT)
wind  = np.array(WND)

# Número de estados deseados
Nc     = 3 # Número de estados

" Se entrena el HMM y se estima la serie de estados probables"
wind          = wind.reshape(-1,1)
model         = GaussianHMM(n_components=Nc, covariance_type="diag", n_iter=1000).fit(wind)
hidden_states = model.predict(wind)

" Matriz de estados, donde cada fila es un año de estados"
state_matrix = np.reshape(hidden_states, (len(normal_years), len(hidden_states)/len(normal_years)))
state_matrix = state_matrix + 1 # Para que los estados empiecen desde 1, y no desde 0

#state_matrix[state_matrix==4] == 33
#state_matrix[state_matrix==3] == 44

#state_matrix[state_matrix==33] == 3
#state_matrix[state_matrix==44] == 4

"Ploteando con pcolor matriz de Viterbi de Estados"
# Dos estados
if Nc == 2:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.)]
	bounds = np.array([0.5, 1.5, 2.5])

# Tres estados
if Nc == 3:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5])

# Cuatro estados
if Nc == 4:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

# Cinco estados
if Nc == 5:
	Colors = [(255/255., 96/255., 61/255.), (67/255., 173/255., 147/255.), (255/255., 195/255., 0), (88/255., 24/255., 69/255.),
			  (11/255., 83/255., 69/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])

cmap_name = 'my_list'
cm        = colors.LinearSegmentedColormap.from_list(cmap_name, Colors, N=Nc) 

fig = plt.figure(figsize=(18,10))
ax  = fig.add_axes([0.06, 0.05, 0.93, 0.8])

norm = colors.BoundaryNorm(boundaries=bounds, ncolors=Nc)
pc  = ax.pcolor(state_matrix, norm=norm, cmap=cm)
cb  = fig.colorbar(pc, orientation='horizontal', ticks=np.arange(Nc)+1, shrink=0.25, pad=0.08)
cb.set_label('State', size=17)
cb.ax.tick_params (labelsize=15)
cb.ax.set_position([0.15, -0.12, 0.2, 0.3])

for col in range(1, state_matrix.shape[1]+1):
	plt.axvline(x=col, ls='-', color='k', lw=1, alpha=1)
	#if col == 106: # 16 de Abril
	#	plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 181: # 30 de Junio
	#	plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 228: # 16 de Agosto
	#	plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)
	#if col == 304: # 31 de Octubre
	#	plt.axvline(x=col, ls='-', color='k', lw=3, alpha=1)

for row in range(1, state_matrix.shape[0]+1):
	plt.axhline(y=row, ls='-', color='k', lw=1, alpha=1)

x_ticks = np.arange(0, state_matrix.shape[1]+1)
y_ticks = np.arange(0, state_matrix.shape[0]+1)

my_y_ticks = ['1979', '1981', '', '', '1895', '', '', '1989', '', '', '1993', '', '', '1997', '', '', '2001', '', '', '2005', '', '', 
              '2009', '', '', '2013', '', '']

my_x_ticks = ['Jan-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Feb-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Mar-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Apr-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'May-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 
		      'Jun-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Jul-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Aug-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Sep-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Oct-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Nov-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Dic-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

plt.xticks(x_ticks, my_x_ticks, size=14)
plt.ylabel('Year', size=17)
plt.yticks(y_ticks, my_y_ticks, size=14)
plt.xlabel('Day', size=17)
ax.set_title(str(Nc)+' States - Wind in PN', fontsize=18)

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/15_expo_2018/Viterbi_matrix_PN_EneDic_'+str(Nc)+'st.png', bbox_inches='tight', dpi=300)

if Nc == 3: #Para hacer los composites	
	punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/StimateEstados_HHM3st_EneDic.bin', 'wb')
	pickle.dump(state_matrix, punto_bin)

	punto_bin = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/StimateEstados_HHM3st_EneDic.bin', 'wb')
	pickle.dump(state_matrix, punto_bin)








