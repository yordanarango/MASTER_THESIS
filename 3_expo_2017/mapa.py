# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:37:59 2017

@author: yordan
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as pl
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
from numpy import genfromtxt

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y)) 


V = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_19880131_V02.0_L3.0_RSS.nc')['vwnd'][0, 314:411, 1019:1140]
U = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_19880131_V02.0_L3.0_RSS.nc')['uwnd'][0, 314:411, 1019:1140]
spd = np.sqrt(U*U + V*V)

Lat = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_19880131_V02.0_L3.0_RSS.nc')['latitude'][314:411]
Lon = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_19880131_V02.0_L3.0_RSS.nc')['longitude'][1019:1140]-360


line_lat = [16.125, 6.125]
line_lon = [-96.125, -78.125]

lat1_TT = 16.125
lat2_TT = 13.375
lon1_TT = -96.125
lon2_TT = -91.125
lat2_PP = 9.125
lon2_PP = -83.625
lat1_PN = 6.125
lon1_PN = -78.125

fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
ax = fig.add_axes([0.1,0.1,0.8,0.8])

map = Basemap(projection='merc', llcrnrlat=0.125, urcrnrlat=24.125, llcrnrlon=-105.125, urcrnrlon=-75.125, resolution='i')
map.drawcoastlines(linewidth = 0.8)
map.drawcountries(linewidth = 0.8)
map.drawparallels(np.arange(0, 24, 8),labels=[1,0,0,1])
map.drawmeridians(np.arange(-105,-75,10),labels=[1,0,0,1])

lons,lats = np.meshgrid(Lon,Lat)
x,y = map(lons,lats)
L_lon, L_lat = map(line_lon, line_lat)
P1_TT_lon, P1_TT_lat = map(lon1_TT, lat1_TT) 
P2_TT_lon, P2_TT_lat = map(lon2_TT, lat2_TT)
P2_PP_lon, P2_PP_lat = map(lon2_PP, lat2_PP)
P1_PN_lon, P1_PN_lat = map(lon1_PN, lat1_PN)
bounds=np.linspace( np.min(spd) ,np.max(spd), 20) 
bounds=np.around(bounds, decimals=2) 
CF = map.contourf(x,y,spd, 20, norm=MidpointNormalize(midpoint=0), cmap=plt.cm.rainbow)#plt.cm.rainbow, plt.cm.RdYlBu_r
cb = plt.colorbar(CF, orientation='horizontal', pad=0.05, shrink=0.8, boundaries=bounds)
cb.set_label('m/s')
Q = map.quiver(x[::2,::2], y[::2,::2], U[::2,::2], V[::2,::2], scale=150)
plt.quiverkey(Q, 0.93, 0.05, 10, '10 m/s' )
ax.set_title('$Wind$ $Field$ $-$ $Winter$ $Anomalies$', size='15')
map.plot(L_lon, L_lat, marker=None, color='k')
map.plot(P1_TT_lon, P1_TT_lat, marker='D', color='k')     
map.plot(P2_TT_lon, P2_TT_lat, marker='D', color='k')    
map.plot(P2_PP_lon, P2_PP_lat, marker='D', color='k')
map.plot(P1_PN_lon, P1_PN_lat, marker='D', color='k')
map.fillcontinents(color='white')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/mapa.png', bbox_inches='tight', dpi=300)


#========================================
"HOVMOLLER"
#========================================

ano    = 2008
wnd_tehuantepec = genfromtxt('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/VIENTOS_RASI/tehuantepec_wind_data.txt', delimiter=',')
wnd_papagayo    = genfromtxt('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/VIENTOS_RASI/papagayo_wind_data.txt', delimiter=',')
wnd_panama      = genfromtxt('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/VIENTOS_RASI/panama_wind_data.txt', delimiter=',')  
fechas_RASI     = pd.date_range('1998/01/01', freq='6H', periods=17532)
TT_RASI         = pd.Series(index = fechas_RASI, data=wnd_tehuantepec)[str(ano)+'-11-01':str(ano+1)+'-03-31']
PP_RASI         = pd.Series(index = fechas_RASI, data=wnd_papagayo)[str(ano)+'-11-01':str(ano+1)+'-03-31']
PN_RASI         = pd.Series(index = fechas_RASI, data=wnd_panama)[str(ano)+'-11-01':str(ano+1)+'-03-31']

fechas = pd.date_range(str(ano)+'-11-01', str(ano+1)+'-03-31', freq='D')

v_pru = np.zeros((len(fechas)*4, 41))
u_pru = np.zeros((len(fechas)*4, 41))

for i, date in enumerate(fechas):
	day   = '%02d' % (date.day,)
	month = '%02d' % (date.month,)
	year  = str(date.year)
	for j in range(41):
		# v_pru[i*4:4*(i+1), j] = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_'+year+month+day+'_V02.0_L3.0_RSS.nc')['vwnd'][:, 378-j, 1055+2*j]
		# u_pru[i*4:4*(i+1), j] = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/CCMP/CCMP_Wind_Analysis_'+year+month+day+'_V02.0_L3.0_RSS.nc')['uwnd'][:, 378-j, 1055+2*j]

		v_pru[i*4:4*(i+1), j] = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/'+year+'/CCMP_Wind_Analysis_'+year+month+day+'_V02.0_L3.0_RSS.nc')['vwnd'][:, 378-j, 1055+2*j]
		u_pru[i*4:4*(i+1), j] = nc.Dataset('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/DATOS/CCMP/'+year+'/CCMP_Wind_Analysis_'+year+month+day+'_V02.0_L3.0_RSS.nc')['uwnd'][:, 378-j, 1055+2*j]


spd_pru = np.sqrt(v_pru*v_pru+u_pru*u_pru)


for i, j in enumerate(TT_RASI):
	if j == 0:
		spd_pru[i, :12] = 0

for i, j in enumerate(PP_RASI):
	if j == 0:
		spd_pru[i, 12:12+18] = 0

for i, j in enumerate(PN_RASI):
	if j == 0:
		spd_pru[i, 12+18:] = 0

time = np.arange(len(fechas)*4)
pos  = np.arange(41)

x, y = np.meshgrid(pos, time)

fig = plt.figure(figsize=(5,6), edgecolor='W',facecolor='W')
ax1 = fig.add_axes([0.0,0.1,0.8,0.8])

cd = ax1.contourf(x, y, spd_pru, np.linspace(6.5,20, 20), cmap=plt.cm.rainbow)
ce = plt.colorbar(cd, drawedges = 'True',format='%.1f')
ce.set_label('m/s')

my_yticks = pd.date_range(str(ano)+'-11-01', str(ano+1)+'-03-31', freq='15D').strftime('%Y-%m-%d')
plt.yticks([0*4,15*4,30*4,45*4,60*4,75*4,90*4,105*4,120*4,135*4,150*4], my_yticks) #0,12,24,36,48,60

my_xticks = ['$TT$ $(15N,$ $94.75W)$','$PP$ $(11N,$ $86.75W)$','$PN$ $(7.5N,$ $79.75W)$']
plt.xticks((4,20,36), my_xticks, size=9)
plt.grid(True)

ax1.set_title('Hovmoller Vientos '+ str(ano) +'-'+str(ano+1), size='12')
ax1.set_xlabel('Posicion', size='10')
ax1.set_ylabel('Tiempo', size='10')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/tercera_exposicion_2017/hovmoller_temporada_'+str(ano)+'.png', bbox_inches='tight', dpi=300)


