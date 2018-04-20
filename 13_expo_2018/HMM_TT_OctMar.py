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
#import seaborn as sns

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
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365
print 'Corregir valor de latitudes. Se está restando 365 en vez de 360. Con ese error están las gráficas'

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[::4] # Se toma una sóla hora del día de la velocidad

"Viento"
pos_lon = np.where(lon == -95)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == 15)[0][0]
v   = archivo['v10'][::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad
u   = archivo['u10'][::4, pos_lat, pos_lon] # Se toma una sóla hora del día de la velocidad
wnd = np.sqrt(v*v+u*u)


"Selección de datos en Octubre, Noviembre, Diciembre, Enero, Febrero, Marzo"
DT  = []
WND = []

leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales

for i in normal_years[1:]-1: # 27 años. Se descartan años bisiestos 		# 80-81
																			# 81-82
	pos = np.where(DATES == pd.Timestamp(str(i)+'-10-01'))[0][0]			# 82-83

																			# 84-85
	DT.append(DATES[pos:pos+182])											# 85-86
	WND.append(wnd[pos:pos+182])											# 86-87

																			# 88-89
dates = pd.DatetimeIndex(np.concatenate(DT))								# 89-90
wind  = np.concatenate(WND)													# 90-91

																			# 92-93
																			# 93-94
																			# 94-95

																			# 96-97
																			# 97-98
																			# 98-99

																			# 00-01
																			# 01-02
																			# 02-03

																			# 04-05
																			# 05-06
																			# 06-07

																			# 08-09
																			# 09-10
																			# 10-11

																			# 12-13
																			# 13-14
																			# 14-15

"Número de estados deseados"
Nc = 5

" Se entrena el HMM y se estima la serie de estados probables"
wind_leap     = wind.reshape(-1,1)
model         = GaussianHMM(n_components=Nc, covariance_type="diag", n_iter=1000).fit(wind_leap)
hidden_states = model.predict(wind_leap)


"Se meten estados en un pickle"

#punto_bin = open('/home/yordan/Escritorio/pickle_hmm.bin', 'wb')
#pickle.dump(hidden_states, punto_bin)

#punto_bin = open('/home/yordan/Escritorio/pickle_hmm.bin', 'wb')
#pickle.dump(hidden_states, punto_bin)

"Se leen estados en un pickle"

#punto_bin = open('/home/yordan/Escritorio/picklehmm.bin', 'rb')
#hidden_states = pickle.load(punto_bin)


" Matriz de estados, donde cada fila es un año de estados"
state_matrix = np.reshape(hidden_states, (27, 182))
state_matrix = state_matrix + 1

state_matrix[state_matrix == 2]  = 44
state_matrix[state_matrix == 4]  = 22
# state_matrix[state_matrix == 2] = 55

state_matrix[state_matrix == 22]  = 2
state_matrix[state_matrix == 44]  = 4
# state_matrix[state_matrix == 55] = 5

# Dos estados
if Nc == 2:
	Colors = [(236/255., 112/255., 99/255.), (125/255., 206/255., 160/255.)]
	bounds = np.array([0.5, 1.5, 2.5])

# Tres estados
if Nc == 3:
	Colors = [(236/255., 112/255., 99/255.), (125/255., 206/255., 160/255.), (155/255., 89/255., 182/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5])

# Cuatro estados
if Nc == 4:
	Colors = [(236/255., 112/255., 99/255.), (125/255., 206/255., 160/255.), (155/255., 89/255., 182/255.), (249/255., 231/255., 159/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

# Cinco estados
if Nc == 5:
	Colors = [(236/255., 112/255., 99/255.), (125/255., 206/255., 160/255.), (155/255., 89/255., 182/255.), (249/255., 231/255., 159/255.),
			  (93/255., 109/255., 126/255.)]
	bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])


"Ploteando con pcolor matriz de Viterbi de Estados"
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

for row in range(1, state_matrix.shape[0]+1):
	plt.axhline(y=row, ls='-', color='k', lw=1, alpha=1)

x_ticks = np.arange(0, state_matrix.shape[1]+1)
y_ticks = np.arange(0, state_matrix.shape[0]+1)

my_y_ticks = ['80-81', '', '', '84-85', '', '', '88-89', '', '', '92-93', '', '', '96-97', '', '', '00-01', '', '', '04-05', '', '',
              '08-09', '', '', '12-13', '', '']

my_x_ticks = ['Oct-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
			  'Nov-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Dic-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Ene-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Feb-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
		      'Mar-01', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']

plt.xticks(x_ticks, my_x_ticks, size=14)
plt.ylabel('Year', size=17)
plt.yticks(y_ticks, my_y_ticks, size=14)
plt.xlabel('Day', size=17)
ax.set_title(str(Nc)+' States - Wind in PN', fontsize=18)

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/Viterbi_matrix_PN_OctMar_'+str(Nc)+'st.png', bbox_inches='tight', dpi=300)
#plt.show()


"########################################       Ploteo de altura geopotencial       ##########################################"

"Selección fechas en las que se dio cada uno de los estados"

# pos_st_1  = np.where(hidden_states == 0)[0]
# pos_st_2  = np.where(hidden_states == 1)[0]
# pos_st_3  = np.where(hidden_states == 2)[0]
# pos_st_4  = np.where(hidden_states == 3)[0]

# date_st_1 = dates[pos_st_1]
# date_st_2 = dates[pos_st_2]
# date_st_3 = dates[pos_st_3]
# date_st_4 = dates[pos_st_4]

# Height = [1000, 850, 300]
# MESES  = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

# for H in Height:
# 	"Datos geopotencial"
# 	G       = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/geopotential/geopotential_'+str(H)+'hPa.nc') #netCDF de altura geopotencial

# 	time_G    = G['time'][:]
# 	cdftime_G = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
# 	fechas_G  = [cdftime_G.num2date(k) for k in time_G]
# 	dates_G   = pd.DatetimeIndex(fechas_G)

# 	lat        = G['latitude'][:]
# 	lon        = G['longitude'][:]-360
# 	lat_inf    = np.where(lat == 0)[0][0]
# 	lat_sup    = np.where(lat == 70)[0][0]
# 	lon_inf    = np.where(lon == -30)[0][0]
# 	lon_sup    = np.where(lon == -190)[0][0]
# 	lat        = lat[lat_sup:lat_inf+1]
# 	lon        = lon[lon_sup:lon_inf+1]


# 	"Si se hace en work station o en cluster, hágase el ciclo anual del ciclo diurno como sigue"
# 	#Geo_pot    = G['z'][:, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8      #extrayendo datos de altura geopotencial
# 	#CICLO      = ciclo_diurno_anual(Geo_pot, dates_G, len(lat), len(lon))

# 	"Si se tiene computadorcito normal, hágase ciclo anual de ciclo diurno así"
# 	CICLO = pickle.load(open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/CICLO_'+str(H)+'hPa.bin', 'rb'))

# 	Anom_G_st1 = np.zeros((len(date_st_1),len(lat),len(lon)))
# 	Anom_G_st2 = np.zeros((len(date_st_2),len(lat),len(lon)))
# 	Anom_G_st3 = np.zeros((len(date_st_3),len(lat),len(lon)))
# 	Anom_G_st4 = np.zeros((len(date_st_4),len(lat),len(lon)))

# 	for i, D in enumerate(date_st_1):
# 		mes           = D.month
# 		hora          = D.hour
# 		clima         = CICLO[MESES[mes-1]+'_'+str(hora)]

# 		POS           = np.where(dates_G == D)[0][0]
# 		"Si no se tiene workstation o cluster"
# 		Anom_G_st1[i] = G['z'][POS, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8 - clima
# 		"Si se tiene workstation o cluster"
# 		#Anom_G_st1[i] = Geo_pot[POS] - clima

# 	mean_anom_st1 = np.mean(Anom_G_st1, axis=0)
# 	min_st1       = np.min(mean_anom_st1)
# 	max_st1       = np.max(mean_anom_st1)

# 	for i, D in enumerate(date_st_2):
# 		mes           = D.month
# 		hora          = D.hour
# 		clima         = CICLO[MESES[mes-1]+'_'+str(hora)]

# 		POS           = np.where(dates_G == D)[0][0]
# 		"Si no se tiene workstation o cluster"
# 		Anom_G_st2[i] = G['z'][POS, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8 - clima
# 		"Si se tiene workstation o cluster"
# 		#Anom_G_st2[i] = Geo_pot[POS] - clima

# 	mean_anom_st2 = np.mean(Anom_G_st2, axis=0)
# 	min_st2       = np.min(mean_anom_st2)
# 	max_st2       = np.max(mean_anom_st2)

# 	for i, D in enumerate(date_st_3):
# 		mes           = D.month
# 		hora          = D.hour
# 		clima         = CICLO[MESES[mes-1]+'_'+str(hora)]

# 		POS           = np.where(dates_G == D)[0][0]
# 		"Si no se tiene workstation o cluster"
# 		Anom_G_st3[i] = G['z'][POS, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8 - clima
# 		"Si se tiene workstation o cluster"
# 		#Anom_G_st3[i] = Geo_pot[POS] - clima

# 	mean_anom_st3 = np.mean(Anom_G_st3, axis=0)
# 	min_st3       = np.min(mean_anom_st3)
# 	max_st3       = np.max(mean_anom_st3)

# 	for i, D in enumerate(date_st_4):
# 		mes           = D.month
# 		hora          = D.hour
# 		clima         = CICLO[MESES[mes-1]+'_'+str(hora)]

# 		POS           = np.where(dates_G == D)[0][0]
# 		"Si no se tiene workstation o cluster"
# 		Anom_G_st4[i] = G['z'][POS, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8 - clima
# 		"Si se tiene workstation o cluster"
# 		#Anom_G_st4[i] = Geo_pot[POS] - clima

# 	mean_anom_st4 = np.mean(Anom_G_st4, axis=0)
# 	min_st4       = np.min(mean_anom_st4)
# 	max_st4       = np.max(mean_anom_st4)

# 	MIN = np.min([min_st1, min_st2, min_st3, min_st4])
# 	MAX = np.max([max_st1, max_st2, max_st3, max_st4])


# 	Ttl1  = 'Composites of Geopotential Height Anomalies \n State 1 - '+str(H)+' hPa'
# 	path1 = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/'+str(H)+'/st1_'+str(H)+'hPa'
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st1, MIN, MAX, u'm', Ttl1, path1, C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st1, min_st1, max_st1, u'm', Ttl1, path1+'_distinct_cb', C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	print H, 'State 1'

# 	Ttl2  = 'Composites of Geopotential Height Anomalies \n State 2 - '+str(H)+' hPa'
# 	path2 = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/'+str(H)+'/st2_'+str(H)+'hPa'
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st2, MIN, MAX, u'm', Ttl2, path2, C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st2, min_st2, max_st2, u'm', Ttl2, path2+'_distinct_cb', C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	print H, 'State 2'

# 	Ttl3  = 'Composites of Geopotential Height Anomalies \n State 3 - '+str(H)+' hPa'
# 	path3 = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/'+str(H)+'/st3_'+str(H)+'hPa'
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st3, MIN, MAX, u'm', Ttl3, path3, C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st3, min_st3, max_st3, u'm', Ttl3, path3+'_distinct_cb', C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	print H, 'State 3'

# 	Ttl4  = 'Composites of Geopotential Height Anomalies \n State 4 - '+str(H)+' hPa'
# 	path4 = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/'+str(H)+'/st4_'+str(H)+'hPa'
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st4, MIN, MAX, u'm', Ttl4, path4, C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	plotear(lat[-1], lat[0], lon[0], lon[-1], 20, 20, lon, lat, mean_anom_st4, min_st4, max_st4, u'm', Ttl4, path4+'_distinct_cb', C_T='k', wind=False, mapa_u=None, mapa_v=None)
# 	print H, 'State 4'
