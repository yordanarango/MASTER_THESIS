import numpy as np
import pandas as pd
import math
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt
from windrose import WindroseAxes

def direction(U, V):
	"U: x wind component serie of numpy"
	"V: y wind component serie of numpy"

	#return
	"d: wind direction serie of numpy"

	d = []

	for u, v in zip(U, V):

		if u > 0 and v > 0:
			#print (np.arctan(u/v)*180/math.pi)
			D = (np.arctan(u/v)*180/math.pi)
			d.append(D)
		
		elif u > 0 and v < 0:
			#print (np.arctan(u/v)*180/math.pi) + 360
			D = (np.arctan(u/v)*180/math.pi) + 360
			d.append(D)

		elif u < 0 and v > 0:
			#print (np.arctan(u/v)*180/math.pi) + 180
			D = (np.arctan(u/v)*180/math.pi) + 180
			d.append(D)

		elif u < 0 and v < 0:
			#print (np.arctan(u/v)*180/math.pi) + 180
			D = (np.arctan(u/v)*180/math.pi) + 180
			d.append(D)

		elif u > 0 and v == 0:
			D = 0.
			d.append(D)

		elif u < 0 and v == 0:
			D = 180.
			d.append(D)

		elif u == 0 and v > 0:
			D = 90.
			d.append(D)

		elif u == 0 and v < 0:
			D = 270.
			d.append(D)

	return np.array(d)

"Lectura datos netCDF de viento"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365


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


"Selección de datos en Noviembre, Diciembre, Enero, Febrero, Marzo"
DT  = [] 
WND = []
U   = []
V   = []


leap_years   = np.array([x for x in set(DATES[DATES.is_leap_year].year)])   # Años bisiestos
normal_years = np.array([x for x in set(DATES[~DATES.is_leap_year].year)])  # Años normales

for i in normal_years[1:]-1: # 27 años. Se descartan años bisiestos 	
																			
	pos = np.where(DATES == pd.Timestamp(str(i)+'-11-01'))[0][0]			
																			
	DT.append(DATES[pos:pos+151])											
	WND.append(wnd[pos:pos+151])											
	U.append(u[pos:pos+151])
	V.append(v[pos:pos+151])												
																			
dates = pd.DatetimeIndex(np.concatenate(DT))								
wind  = np.concatenate(WND)
U     = np.concatenate(U)
V     = np.concatenate(V)


"Rosa de viento de Noviembre a Marzo"
Dir   = direction(U, V)
ax = WindroseAxes.from_ax()
ax.bar(Dir, wind, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/rosa_NOV_MAR.png', bbox_inches='tight', dpi=300)
plt.close('all')


"Rosa de viento todo el año"
Dir_year = direction(u, v)
ax       = WindroseAxes.from_ax()
ax.bar(Dir_year, wnd, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/rosa_all_year.png', bbox_inches='tight', dpi=300)
plt.close('all')
