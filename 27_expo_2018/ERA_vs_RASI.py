# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 04:33:14 2018

@author: yordan
"""

import numpy as np
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta


"##################################     FUNCIONES    ####################################"
def indices(Serie): # Devuelve posiciones donde empieza y termina un evento
	# Serie que contiene dato cuando hay evento de chorro, y ceros cuando no se considera que hay evento de chorro

	EVENTOS = []
	i = 0
	while i < len(Serie)-1:

		if Serie[i] != 0:
			aux = i
			while Serie[aux+1] != 0:
				aux = aux + 1
				if aux == len(Serie)-1:
					break
			fin = aux

			EVENTOS.append([i, fin])

			i = aux + 2 # Siguiente posicion a examinar
		else:
			i = i + 1 # Siguiente posicion a examinar

	return np.array(EVENTOS)



"#########################################################################################"


"DATOS RASI"


RASI_Tehuantepec = []
RASI_Papagayo    = []
RASI_Panama      = []

for i in range(1998, 2012):

	mean_TT = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Tehuantepec_'+str(i)+'.nc')
	mean_PP = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Papagayo_'+str(i)+'.nc')
	mean_PN = nc.Dataset(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/RASI/RASI_Panama_'+str(i)+'.nc')

	RASI_Tehuantepec.extend(mean_TT['WindSpeedMean'][:])
	RASI_Papagayo.extend(mean_PP['WindSpeedMean'][:])
	RASI_Panama.extend(mean_PN['WindSpeedMean'][:])

	mean_TT.close()
	mean_PP.close()
	mean_PN.close()


EVN_TT     = indices(RASI_Tehuantepec)
EVN_PP     = indices(RASI_Papagayo)
EVN_PN     = indices(RASI_Panama)


"FECHAS DE CADA EVENTO"

Dates_RASI   = pd.date_range('1998-01-01', freq='6H', periods=len(RASI_Tehuantepec))

Dates_TT_ini = [str(Dates_RASI[x[0]]) for x in EVN_TT] # Fechas en las que empezaron los eventos
Dates_PP_ini = [str(Dates_RASI[x[0]]) for x in EVN_PP] # Fechas en las que empezaron los eventos
Dates_PN_ini = [str(Dates_RASI[x[0]]) for x in EVN_PN] # Fechas en las que empezaron los eventos

Dates_TT_fin = [str(Dates_RASI[x[1]]) for x in EVN_TT] # Fechas en las que terminaron los eventos
Dates_PP_fin = [str(Dates_RASI[x[1]]) for x in EVN_PP] # Fechas en las que terminaron los eventos
Dates_PN_fin = [str(Dates_RASI[x[1]]) for x in EVN_PN] # Fechas en las que terminaron los eventos



"DATOS ERA-INTERIM"
file = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/ERA-INTERIM/WIND/wind_big_area.nc')

lat_ERA     = file['latitude'][:]
lon_ERA     = file['longitude'][:] - 360

time_ERA    = file['time'][:]
cdftime_ERA = utime('hours since 1900-01-01 00:00:0.0', calendar='gregorian')
dates       = [cdftime_ERA.num2date(x) for x in time_ERA]
dts_ERA     = pd.DatetimeIndex(dates)


"BOX LIMITS AND POINT"
box_TT_lon = [-93.75, -96.25]; box_TT_lat = [16    ,  12.5]
box_PP_lon = [-86.0 , -90.5] ; box_PP_lat = [11.75 ,  8.75]
box_PN_lon = [-78.75, -80.75]; box_PN_lat = [8.5   ,  5.5]

p_TT_lon = -95.0  ; p_TT_lat =  14.25
p_PP_lon = -88.25 ; p_PP_lat =  10.25
p_PN_lon = -80.0  ; p_PN_lat =  6.25


"GRAFICAS"

for ch in ['TT', 'PP', 'PN']:

	if ch == 'TT':
		box_lat = box_TT_lat
		box_lon = box_TT_lon

		p_lat   = p_TT_lat
		p_lon   = p_TT_lon

		Dates_ini = Dates_TT_ini 
		Dates_fin = Dates_TT_fin 

	elif ch == 'PP':
		box_lat = box_PP_lat
		box_lon = box_PP_lon

		p_lat   = p_PP_lat
		p_lon   = p_PP_lon

		Dates_ini = Dates_PP_ini 
		Dates_fin = Dates_PP_fin 

	elif ch == 'PN':
		box_lat = box_PN_lat
		box_lon = box_PN_lon

		p_lat   = p_PN_lat
		p_lon   = p_PN_lon

		Dates_ini = Dates_PN_ini 
		Dates_fin = Dates_PN_fin

	"Serie completa de speed de ERA"

	N_lim_box_ERA = np.where(lat_ERA == box_lat[0])[0][0] # northern's limit position of box 
	S_lim_box_ERA = np.where(lat_ERA == box_lat[1])[0][0] # southern's limit position of box
	E_lim_box_ERA = np.where(lon_ERA == box_lon[0])[0][0] # easthern's limit position of box
	W_lim_box_ERA = np.where(lon_ERA == box_lon[1])[0][0] # westhern's limit position of box

	p_lat_pos_ERA = np.where(lat_ERA == p_lat)[0][0] # position in lat of point
	p_lon_pos_ERA = np.where(lon_ERA == p_lon)[0][0] # position in lon of point

	serie_box_u_ERA = file['u10'][:, N_lim_box_ERA : S_lim_box_ERA + 1 , W_lim_box_ERA : E_lim_box_ERA + 1] # u serie of box
	serie_p_u_ERA   = file['u10'][:, p_lat_pos_ERA , p_lon_pos_ERA]                                         # u serie of point

	serie_box_v_ERA = file['v10'][:, N_lim_box_ERA : S_lim_box_ERA + 1 , W_lim_box_ERA : E_lim_box_ERA + 1] # v serie of box
	serie_p_v_ERA   = file['v10'][:, p_lat_pos_ERA , p_lon_pos_ERA]                                         # v serie of point

	serie_box_ERA   = np.sqrt(serie_box_u_ERA * serie_box_u_ERA + serie_box_v_ERA * serie_box_v_ERA)        # spd serie of box
	
	SERIE_BOX_ERA   = []
	for i in range(len(serie_box_ERA)):                                      # speed index serie in box
		SERIE_BOX_ERA.append(np.mean(serie_box_ERA[i])) 

	SERIE_BOX_ERA   = np.array(SERIE_BOX_ERA)
	SERIE_P_ERA     = np.sqrt(serie_p_u_ERA * serie_p_u_ERA + serie_p_v_ERA * serie_p_v_ERA) # speed index serie in point


	for d_i, d_f in zip(Dates_ini, Dates_fin):                 
		pos_ini_ERA = np.where(dts_ERA == pd.Timestamp(d_i))[0][0] # position in dts_ERA of initial date of each event
		pos_fin_ERA = np.where(dts_ERA == pd.Timestamp(d_f))[0][0] # position in dts_ERA of final date of each event


		fechas    = pd.date_range(pd.Timestamp(d_i) - relativedelta(days=1), pd.Timestamp(d_f) + relativedelta(days=1), freq='6H')

		"extrayendo datos de evento de CCMP"
		
		BOX_ERA   = SERIE_BOX_ERA[pos_ini_ERA - 4 : pos_fin_ERA + 4 + 1] 
		POINT_ERA = SERIE_P_ERA[pos_ini_ERA - 4 : pos_fin_ERA + 4 + 1]

		
		"extrayendo datos de evento de CCMP"

		SERIE_BOX_CCMP   = []
		SERIE_P_CCMP     = []

		for dts in fechas:
			d_CCMP = dts.day
			m_CCMP = dts.month
			y_CCMP = dts.year
			h_CCMP = dts.hour

			file_CCMP = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/' + str(y_CCMP) + '/CCMP_Wind_Analysis_' + str(y_CCMP) + '%02d' % m_CCMP  + '%02d' % d_CCMP + '_V02.0_L3.0_RSS.nc')
			
			lat_CCMP  = file_CCMP['latitude'][:]
			lon_CCMP  = file_CCMP['longitude'][:] - 360

			time_CCMP    = file_CCMP['time'][:]
			cdftime_CCMP = utime('hours since 1987-01-01 00:00:00', calendar='gregorian')
			dts_CCMP     = [cdftime_CCMP.num2date(x) for x in time_CCMP]
			dts_CCMP     = pd.DatetimeIndex(dts_CCMP)

			N_lim_box_CCMP = np.where(lat_CCMP == box_lat[0] - 0.125)[0][0] # northern's limit position of box. Hay que restar 0.125 dado que el vector lat_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875 
			S_lim_box_CCMP = np.where(lat_CCMP == box_lat[1] - 0.125)[0][0] # southern's limit position of box. Hay que restar 0.125 dado que el vector lat_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875
			E_lim_box_CCMP = np.where(lon_CCMP == box_lon[0] - 0.125)[0][0] # easthern's limit position of box. Hay que restar 0.125 dado que el vector lon_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875
			W_lim_box_CCMP = np.where(lon_CCMP == box_lon[1] - 0.125)[0][0] # westhern's limit position of box. Hay que restar 0.125 dado que el vector lon_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875

			p_lat_pos_CCMP = np.where(lat_CCMP == p_lat - 0.125)[0][0] # position in lat of point. Hay que restar 0.125 dado que el vector lat_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875
			p_lon_pos_CCMP = np.where(lon_CCMP == p_lon - 0.125)[0][0] # position in lon of point. Hay que restar 0.125 dado que el vector lon_CCMP tiene decimales 0.125, 0.375, 0.625 y 0.875

			data_box_u_CCMP = file_CCMP['uwnd'][h_CCMP//6, S_lim_box_CCMP : N_lim_box_CCMP + 1 , W_lim_box_CCMP : E_lim_box_CCMP + 1] # u serie of box
			data_p_u_CCMP   = file_CCMP['uwnd'][h_CCMP//6, p_lat_pos_CCMP , p_lon_pos_CCMP]                                           # u serie of point

			data_box_v_CCMP = file_CCMP['vwnd'][h_CCMP//6, S_lim_box_CCMP : N_lim_box_CCMP + 1 , W_lim_box_CCMP : E_lim_box_CCMP + 1] # v serie of box
			data_p_v_CCMP   = file_CCMP['vwnd'][h_CCMP//6, p_lat_pos_CCMP , p_lon_pos_CCMP]                                           # v serie of point

			data_box_CCMP   = np.sqrt(data_box_u_CCMP * data_box_u_CCMP + data_box_v_CCMP * data_box_v_CCMP)                 # spd serie of box

			SERIE_BOX_CCMP.append(np.mean(data_box_CCMP))
			SERIE_P_CCMP.append(np.sqrt(data_p_u_CCMP * data_p_u_CCMP + data_p_v_CCMP * data_p_v_CCMP))

			file_CCMP.close()

		BOX_CCMP   = np.array(SERIE_BOX_CCMP)
		POINT_CCMP = np.array(SERIE_P_CCMP)


		"PLOT"

		plt.figure(figsize=(10,7))

		plt.plot(BOX_ERA, label='ERA-Int Box', color='k')      # box serie ERA
		plt.plot(POINT_ERA, label='ERA-Int Point', color='k', alpha=0.7)  # point serie ERA
		plt.plot(BOX_CCMP, label='CCMP Box', color='#cb4335')     # box serie CCMP
		plt.plot(POINT_CCMP, label='CCMP Point', color='#cb4335', alpha=0.7) # point serie CCMP

		Xaxis    = fechas.strftime('%m-%d')
		
		plt.axvline(x=np.arange(len(Xaxis))[4], color='b')
		plt.axvline(x=np.arange(len(Xaxis))[-5], color='b')

		plt.xticks(np.arange(len(Xaxis)), Xaxis)
		
		plt.xlabel('dates (' + str(fechas.year[0]) + ')', fontsize=15)
		plt.ylabel('velocity (m/s)', fontsize=15)

		#plt.grid(True)
		plt.legend(loc='best')
		plt.title('Event of gap wind - ' + ch, fontsize=15)
		
		plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/ERA_vs_RASI/' + ch +'/'+ d_i[:10] + '.png', bbox_inches='tight', dpi=300)
		plt.close('all')

file.close()

"AREAS Y PUNTOS DE DONDE SE EXTRAJERON LAS SERIES PARA COMPARAR"


# p_TT_lon = -95.0
# p_PP_lon = -88.25
# p_PN_lon = -80.0
# p_TT_lat = 14.25
# p_PP_lat = 10.25
# p_PN_lat = 6.25

# box_TT_lon = [-96.25, -96.25, -93.75, -93.75, -96.25]
# box_TT_lat = [16, 12.5, 12.5, 16, 16]
# box_PP_lon = [-90.5, -90.5, -86.0, -86.0, -90.5]
# box_PP_lat = [11.75, 8.75, 8.75, 11.75, 11.75]
# box_PN_lon = [-80.75, -80.75, -78.75, -78.75, -80.75]
# box_PN_lat = [8.5, 5.5, 5.5, 8.5, 8.5]

# fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
# ax = fig.add_axes([0.1,0.1,0.8,0.8])

# map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=24, llcrnrlon=-105, urcrnrlon=-75, resolution='i')
# map.drawcoastlines(linewidth = 0.8)
# map.drawcountries(linewidth = 0.8)
# map.drawparallels(np.arange(0, 24, 8),labels=[1,0,0,1])
# map.drawmeridians(np.arange(-105,-75,10),labels=[1,0,0,1])


# P_TT_lon, P_TT_lat = map(p_TT_lon, p_TT_lat) 
# P_PP_lon, P_PP_lat = map(p_PP_lon, p_PP_lat)  
# P_PN_lon, P_PN_lat = map(p_PN_lon, p_PN_lat)   
# TT_lon,TT_lat = map(box_TT_lon, box_TT_lat)
# PP_lon,PP_lat = map(box_PP_lon, box_PP_lat)
# PN_lon,PN_lat = map(box_PN_lon, box_PN_lat)


# map.plot(P_TT_lon, P_TT_lat, marker='D', color='k')
# map.plot(P_PP_lon, P_PP_lat, marker='D', color='k')
# map.plot(P_PN_lon, P_PN_lat, marker='D', color='k')
# map.plot(TT_lon, TT_lat, marker=None, color='k')
# map.plot(PP_lon, PP_lat, marker=None, color='k')
# map.plot(PN_lon, PN_lat, marker=None, color='k')      

# plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/ubicacion_series.png', bbox_inches='tight', dpi=300)
# plt.close('all')



