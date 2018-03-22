import numpy as np
import netCDF4 as nc
from netcdftime import utime
import matplotlib.pyplot as plt
import pandas as pd
from dateutil.relativedelta import relativedelta
import os
import mpl_toolkits.axisartist as AA
from mpl_toolkits.axes_grid1 import host_subplot
from mpl_toolkits.basemap import Basemap


archivo = nc.Dataset('/home/yordan/Escritorio/roms_ini.nc')
variables = [x for x in archivo.variables]
temp = archivo.variables['temp'][0,0]

plt.imshow(temp)
plt.colorbar()
plt.show()




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
ax.set_title(titulo, size='15', color = C_T)

if wind == True:
	Q = map.quiver(x[::2,::2], y[::2,::2], mapa_u[::2,::2], mapa_v[::2,::2], scale=150)
	plt.quiverkey(Q, 0.93, 0.05, 10, '10 m/s' )

#map.fillcontinents(color='white')
plt.savefig(path+'.png', bbox_inches='tight', dpi=300)
plt.close('all')