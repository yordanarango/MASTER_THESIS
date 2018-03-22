# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 13:34:25 2018

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
def ciclo_diurno_anual(matriz, fechas, len_lat, len_lon):
	#matriz : matriz de numpy de 3 dimensiones donde cada capa corresponde a una fecha en el vector de pandas "fechas"
	#fechas : objeto de pandas con las fechas que corresponden a cada una de las capas en matríz
	#len_lat: integer cantidad de pixeles en direccion meridional
	#len_lon: integer cantidad de pixeles en direccion zonal

	#return: devuelve diccionario con ciclo diuno para cada mes
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

"Se leen datos de viento a resolución de 0.25 grados"
# archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')
# lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365

"Fechas"
# time    = archivo['time'][:]
# cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
# fechas  = [cdftime.num2date(x) for x in time]
# DATES   = pd.DatetimeIndex(fechas)[:]

"Viento"
# v   = archivo['v10'][:] # Para quedar con las mismas fechas del archivo U y V, 6 horas, 10 m, 1979-2016.nc, con el que se hizo
# u   = archivo['u10'][:] # Para quedar con las mismas fechas del archivo U y V, 6 horas, 10 m, 1979-2016.nc, con el que se hizo
# wnd = np.sqrt(v*v+u*u)

"Se calcula ciclo anual de ciclo diurno"
# CICLO_WIND = ciclo_diurno_anual(wnd, DATES, len(lat), len(lon))

"Si no se tiene buen computador, léase dictionario con los ciclos que ya se han calculado anteriormente"
a = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/ciclo_diurno_anual_wind_025_6h.bin', 'rb')
b = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/ciclo_diurno_anual_U_025_6h.bin', 'rb')
c = open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/ciclo_diurno_anual_V_025_6h.bin', 'rb')

CICLO_WIND = pickle.load(a)
CICLO_U    = pickle.load(b)
CICLO_V    = pickle.load(c)

"##################################################      COMPUESTOS        ##########################################################"

"Se lee matriz de estados"
state_matrix_2st = pickle.load(open('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/StimateEstados_HHM2st.bin','rb'))


"Se leen datos de viento a resolución de 0.25 grados"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/UyV_1979_2016_res025.nc')


"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)[3::4]
DATES   = DATES[:-61] # Para quedar con las mismas fechas del archivo U y V, 6 horas, 10 m, 1979-2016.nc, con el que se hizo

"Se extraen fechas en estado 2, para un modelo con dos estados"
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)]) #Años normales. NO bisiestos

DT_st2          = [] 
year_cualquiera = pd.date_range('2001-01-01', '2001-12-31', freq='D')   #año cualquiera para poder estraer mes y día
for i, d in enumerate(normal_years):
	for j in range(365):
		if state_matrix_2st[i, j] == 2:
			DT_st2.append(str(d)+'-'+str(year_cualquiera[j].month)+'-'+str(year_cualquiera[j].day)+' -18:00:00')

DT = pd.DatetimeIndex(DT_st2) #Vuelvo fechas de pandas


"Se seleccionan fechas entre Noviembre 1 a Diciembre 31 y de Enero 1 a Abril 15, en estado 2"
"Se seleccionan fechas entre Abril 16 y Junio 30, y Agosto 16 y Octubre 31, y en estado 2"

dt_EneAbr = [] #fechas de Enero 1 a Abril 15 en estado dos
dt_NovDic = [] #fechas de Noviembre 1 a Diciembre 31 en estado dos

dt_AbrJun = [] #fechas de abril 16 a junio 30 en estado dos
dt_AugOct = [] #fechas de agosto 16 a octubre 31 en estado dos

for d in normal_years:

	pos_EA = np.where((str(d)+'-01-01' <= DT) & (DT <= str(d)+'-04-15'))[0] # posiciones de fechas de Enero 1 a Abril 15 en estado dos
	pos_ND = np.where((str(d)+'-11-01' <= DT) & (DT <= str(d)+'-12-31'))[0] # posiciones de fechas de Noviembre 1 a Diciembre 31 en estado dos

	pos_AJ = np.where((str(d)+'-04-16' <= DT) & (DT <= str(d)+'-06-30'))[0] # posiciones de fechas de abril 16 a junio 30 en estado dos 
	pos_AO = np.where((str(d)+'-08-16' <= DT) & (DT <= str(d)+'-10-31'))[0] # posiciones de fechas de agosto 16 a octubre 31 en estado dos

	dt_EneAbr.append(DT[pos_EA])
	dt_NovDic.append(DT[pos_ND])

	dt_AbrJun.append(DT[pos_AJ])
	dt_AugOct.append(DT[pos_AO])

Dt_EneAbr = pd.DatetimeIndex(np.concatenate(dt_EneAbr))
Dt_NovDic = pd.DatetimeIndex(np.concatenate(dt_NovDic))

Dt_AbrJun = pd.DatetimeIndex(np.concatenate(dt_AbrJun))
Dt_AugOct = pd.DatetimeIndex(np.concatenate(dt_AugOct))


"Se hacen compuestos para Abril-Junio, y Agosto-Octubre"

lat = archivo.variables['latitude'][:]     # va desde 7°N hasta 25°N
lon = archivo.variables['longitude'][:]-365  # va desde -64.5°W hasta -101°W

time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
Dates   = pd.DatetimeIndex(fechas)[:]

CompU_EneAbr = np.zeros((len(Dt_EneAbr), len(lat), len(lon)))
CompU_NovDic = np.zeros((len(Dt_NovDic), len(lat), len(lon)))

CompV_EneAbr = np.zeros((len(Dt_EneAbr), len(lat), len(lon)))
CompV_NovDic = np.zeros((len(Dt_NovDic), len(lat), len(lon)))

CompU_AbrJun = np.zeros((len(Dt_AbrJun), len(lat), len(lon)))
CompU_AugOct = np.zeros((len(Dt_AugOct), len(lat), len(lon)))

CompV_AbrJun = np.zeros((len(Dt_AbrJun), len(lat), len(lon)))
CompV_AugOct = np.zeros((len(Dt_AugOct), len(lat), len(lon)))

MESES        = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

for i, d in enumerate(Dt_EneAbr):
	pos_date          = np.where(Dates == d)[0][0]
	u                 = archivo.variables['u10'][pos_date]
	v                 = archivo.variables['v10'][pos_date]
	
	mes               = d.month
	
	cc_U              = CICLO_U[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	cc_V              = CICLO_V[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	
	CompU_EneAbr[i]   = u - cc_U
	CompV_EneAbr[i]   = v - cc_V

for i, d in enumerate(Dt_NovDic):
	pos_date          = np.where(Dates == d)[0][0]
	u                 = archivo.variables['u10'][pos_date]
	v                 = archivo.variables['v10'][pos_date]
	
	mes               = d.month

	cc_U              = CICLO_U[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	cc_V              = CICLO_V[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	
	CompU_NovDic[i]   = u - cc_U
	CompV_NovDic[i]   = v - cc_V

for i, d in enumerate(Dt_AbrJun):
	pos_date          = np.where(Dates == d)[0][0]
	u                 = archivo.variables['u10'][pos_date]
	v                 = archivo.variables['v10'][pos_date]
	
	mes               = d.month
	
	cc_U              = CICLO_U[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	cc_V              = CICLO_V[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	
	CompU_AbrJun[i]   = u - cc_U
	CompV_AbrJun[i]   = v - cc_V

for i, d in enumerate(Dt_AugOct):
	pos_date          = np.where(Dates == d)[0][0]
	u                 = archivo.variables['u10'][pos_date]
	v                 = archivo.variables['v10'][pos_date]
	
	mes               = d.month

	cc_U              = CICLO_U[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	cc_V              = CICLO_V[MESES[mes-1]+'_18']    # Porque los estados se hicieron para la hora de las 18 horas que es cuando se da la mayor velocidad en el día, y fue con lo que se hicieron los HMM
	
	CompU_AugOct[i]   = u - cc_U
	CompV_AugOct[i]   = v - cc_V

"Se calcula la velocidad de las anomalías del viento para los estados 2 entre Ene-Abr, Nov-Dic y Nov-Abr. Se plotea"

COMPU_st2_EneAbr = np.mean(CompU_EneAbr, axis = 0); COMPV_st2_EneAbr = np.mean(CompV_EneAbr, axis = 0); COMPWnd_st2_EneAbr = np.sqrt(COMPU_st2_EneAbr*COMPU_st2_EneAbr + COMPV_st2_EneAbr*COMPV_st2_EneAbr); min_EA = np.min(COMPWnd_st2_EneAbr); max_EA = np.max(COMPWnd_st2_EneAbr); Ttl_EA = 'January-April Wind Composites - State 2 (HMM 2)'; path_EA = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/CompWind_EA_2st'
COMPU_st2_NovDic = np.mean(CompU_NovDic, axis = 0); COMPV_st2_NovDic = np.mean(CompV_NovDic, axis = 0); COMPWnd_st2_NovDic = np.sqrt(COMPU_st2_NovDic*COMPU_st2_NovDic + COMPV_st2_NovDic*COMPV_st2_NovDic); min_ND = np.min(COMPWnd_st2_NovDic); max_ND = np.max(COMPWnd_st2_NovDic); Ttl_ND = 'November-December Wind Composites - State 2 (HMM 2)'; path_ND = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/CompWind_ND_2st'

COMPU_st2_AbrJun = np.mean(CompU_AbrJun, axis = 0); COMPV_st2_AbrJun = np.mean(CompV_AbrJun, axis = 0); COMPWnd_st2_AbrJun = np.sqrt(COMPU_st2_AbrJun*COMPU_st2_AbrJun + COMPV_st2_AbrJun*COMPV_st2_AbrJun); min_AJ = np.min(COMPWnd_st2_AbrJun); max_AJ = np.max(COMPWnd_st2_AbrJun); Ttl_AJ = 'April-Junio Wind Composites - State 2 (HMM 2)'   ; path_AJ = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/CompWind_AJ_2st'
COMPU_st2_AugOct = np.mean(CompU_AugOct, axis = 0); COMPV_st2_AugOct = np.mean(CompV_AugOct, axis = 0); COMPWnd_st2_AugOct = np.sqrt(COMPU_st2_AugOct*COMPU_st2_AugOct + COMPV_st2_AugOct*COMPV_st2_AugOct); min_AO = np.min(COMPWnd_st2_AugOct); max_AO = np.max(COMPWnd_st2_AugOct); Ttl_AO = 'August-October Wind Composites - State 2 (HMM 2)'; path_AO = '/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/CompWind_AO_2st'

MIN = np.min([min_AJ, min_AO, min_EA, min_ND])
MAX = np.max([max_AJ, max_AO, max_EA, max_ND])

plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon[::2], lat[::2], COMPWnd_st2_EneAbr[::2, ::2], MIN, MAX, 'm/s', Ttl_EA, path_EA, C_T='k', wind=True, mapa_u=COMPU_st2_EneAbr[::2, ::2], mapa_v=COMPV_st2_EneAbr[::2, ::2])
plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon[::2], lat[::2], COMPWnd_st2_NovDic[::2, ::2], MIN, MAX, 'm/s', Ttl_ND, path_ND, C_T='k', wind=True, mapa_u=COMPU_st2_NovDic[::2, ::2], mapa_v=COMPV_st2_NovDic[::2, ::2]) 

plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon[::2], lat[::2], COMPWnd_st2_AbrJun[::2, ::2], MIN, MAX, 'm/s', Ttl_AJ, path_AJ, C_T='k', wind=True, mapa_u=COMPU_st2_AbrJun[::2, ::2], mapa_v=COMPV_st2_AbrJun[::2, ::2])
plotear(lat[-1], lat[0], lon[0], lon[-1], 4, 7, lon[::2], lat[::2], COMPWnd_st2_AugOct[::2, ::2], MIN, MAX, 'm/s', Ttl_AO, path_AO, C_T='k', wind=True, mapa_u=COMPU_st2_AugOct[::2, ::2], mapa_v=COMPV_st2_AugOct[::2, ::2]) 



