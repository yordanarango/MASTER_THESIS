import numpy as np
import pickle
import pandas as pd
from netcdftime import pandas
import netCDF4 as nc

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

for H in [1000, 850, 700, 500, 300, 200]:
	G       = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/geopotential/geopotential_'+str(H)+'hPa.nc') #netCDF de altura geopotencial

	time_G     = G['time'][:]
	cdftime_G  = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
	fechas_G   = [cdftime_G.num2date(k) for k in time_G]
	dates_G    = pd.DatetimeIndex(fechas_G)
	
	lat        = G['latitude'][:]
	lon        = G['longitude'][:]-360
	lat_inf    = np.where(lat == 0)[0][0]
	lat_sup    = np.where(lat == 70)[0][0]
	lon_inf    = np.where(lon == -30)[0][0]
	lon_sup    = np.where(lon == -190)[0][0]
	lat        = lat[lat_sup:lat_inf+1]
	lon        = lon[lon_sup:lon_inf+1]

	Geo_pot    = G['z'][:, lat_sup:lat_inf+1, lon_sup:lon_inf+1]/9.8      #extrayendo datos de altura geopotencial
	CICLO      = ciclo_diurno_anual(Geo_pot, dates_G, len(lat), len(lon))

	punto_bin  = open('.../CICLO_'+str(H)+'hPa.bin', 'wb')
	pickle.dump(CICLO, punto_bin)

	punto_bin  = open('.../CICLO_'+str(H)+'hPa.bin', 'wb')
	pickle.dump(CICLO, punto_bin)