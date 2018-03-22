import numpy as np
import pandas as pd
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt

"Lectura datos netCDF de viento"
archivo = nc.Dataset('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
Variables = [x for x in archivo.variables]
lat = archivo.variables['latitude'][:]; lon = archivo.variables['longitude'][:]-365

"Fechas"
time    = archivo['time'][:]
cdftime = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
fechas  = [cdftime.num2date(x) for x in time]
DATES   = pd.DatetimeIndex(fechas)

"Viento en un sólo pixel de Panamá"
pos_lon = np.where(lon == -95)[0][0] # Se toma un sólo pixel
pos_lat = np.where(lat == 15)[0][0]

v_00    = archivo['v10'][::4, pos_lat, pos_lon] # Se toma a las 0 horas
u_00    = archivo['u10'][::4, pos_lat, pos_lon] # Se toma a las 0 horas
wnd_00  = np.sqrt(v_00*v_00+u_00*u_00)

v_06    = archivo['v10'][1::4, pos_lat, pos_lon] # Se toma a las 0 horas
u_06    = archivo['u10'][1::4, pos_lat, pos_lon] # Se toma a las 0 horas
wnd_06  = np.sqrt(v_06*v_06+u_06*u_06)

v_12    = archivo['v10'][2::4, pos_lat, pos_lon] # Se toma a las 0 horas
u_12    = archivo['u10'][2::4, pos_lat, pos_lon] # Se toma a las 0 horas
wnd_12  = np.sqrt(v_12*v_12+u_12*u_12)

v_18    = archivo['v10'][3::4, pos_lat, pos_lon] # Se toma a las 0 horas
u_18    = archivo['u10'][3::4, pos_lat, pos_lon] # Se toma a las 0 horas
wnd_18  = np.sqrt(v_18*v_18+u_18*u_18)

ciclo   = [np.mean(wnd_00), np.mean(wnd_06), np.mean(wnd_12), np.mean(wnd_18)]

color   = ['#610B0B', '#B40404', '#DF3A01', '#FF4000', '#DF7401', '#FFBF00',
 		  '#00FFBF', '#00FFFF', '#01A9DB', '#0080FF', '#013ADF', '#0404B4']
mes     = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111)
ax.plot(ciclo, c='k', linewidth=2, label='Typical Daily Cicle')

for i, m, C in zip(range(1,13), mes, color):

	pos_00 = np.where((DATES.month == i) & (DATES.hour == 0))[0]
	pos_06 = np.where((DATES.month == i) & (DATES.hour == 6))[0]
	pos_12 = np.where((DATES.month == i) & (DATES.hour == 12))[0]
	pos_18 = np.where((DATES.month == i) & (DATES.hour == 18))[0]

	V_00    = archivo['v10'][pos_00, pos_lat, pos_lon] # Se toma a las 0 horas
	U_00    = archivo['u10'][pos_00, pos_lat, pos_lon] # Se toma a las 0 horas
	Wnd_00  = np.sqrt(V_00*V_00+U_00*U_00)

	V_06    = archivo['v10'][pos_06, pos_lat, pos_lon] # Se toma a las 0 horas
	U_06    = archivo['u10'][pos_06, pos_lat, pos_lon] # Se toma a las 0 horas
	Wnd_06  = np.sqrt(V_06*V_06+U_06*U_06)

	V_12    = archivo['v10'][pos_12, pos_lat, pos_lon] # Se toma a las 0 horas
	U_12    = archivo['u10'][pos_12, pos_lat, pos_lon] # Se toma a las 0 horas
	Wnd_12  = np.sqrt(V_12*V_12+U_12*U_12)

	V_18    = archivo['v10'][pos_18, pos_lat, pos_lon] # Se toma a las 0 horas
	U_18    = archivo['u10'][pos_18, pos_lat, pos_lon] # Se toma a las 0 horas
	Wnd_18  = np.sqrt(V_18*V_18+U_18*U_18)

	Ciclo   = [np.mean(Wnd_00), np.mean(Wnd_06), np.mean(Wnd_12), np.mean(Wnd_18)]
	ax.plot(Ciclo, linewidth=2, label=m, c=C)

my_xticks = ['00:00', '06:00', '12:00', '18:00']
plt.xticks(range(4), my_xticks)
plt.title('Wind Daily Cycle', size=15)
ax.set_xlabel('Hour', size=13)
ax.set_ylabel('Wind Velocity (m/s)', size=13)
plt.grid(True)
plt.legend(loc='up right', ncol=4)
plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/13_expo_2018/Wind_Daily_Cicle_PN.png', bbox_inches='tight', dpi=300)
