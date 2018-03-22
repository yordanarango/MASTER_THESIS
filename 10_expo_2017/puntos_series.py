# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:59:37 2016

@author: yordan
"""

from mpl_toolkits.basemap import Basemap
import matplotlib.pylab as pl
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy import linalg as la
import pandas as pd

#==============================================================================
"PLOTEO"
#==============================================================================

p_NA_lat   = 30
p_NA_lon   = -100 

p_TT_lon   = -95
p_TT_lat   = 15
p_G_lon    = -94
p_G_lat    = 20

p_PP_lon   = -88
p_PP_lat   = 11
p_PP_C_lon = -82
p_PP_C_lat = 13

p_PN_lon   = -79
p_PN_lat   = 7
p_PN_C_lon = -78
p_PN_C_lat = 11


fig = plt.figure(figsize=(8,8), edgecolor='W',facecolor='W')
ax = fig.add_axes([0.1,0.1,0.8,0.8])
map = Basemap(projection='merc', llcrnrlat=0, urcrnrlat=40, llcrnrlon=-120, urcrnrlon=-55, resolution='i')
map.drawcoastlines(linewidth = 0.8)
map.drawcountries(linewidth = 0.8)
map.drawparallels(np.arange(0, 40, 10),labels=[1,0,0,1])
map.drawmeridians(np.arange(-120,-55,10),labels=[1,0,0,1])

P_NA_lon, P_NA_lat = map(p_NA_lon, p_NA_lat)

P_G_lon, P_G_lat = map(p_G_lon, p_G_lat)
P_TT_lon, P_TT_lat = map(p_TT_lon, p_TT_lat)

P_PP_lon, P_PP_lat = map(p_PP_lon, p_PP_lat)
P_PP_C_lon, P_PP_C_lat = map(p_PP_C_lon, p_PP_C_lat)

P_PN_lon, P_PN_lat = map(p_PN_lon, p_PN_lat)
P_PN_C_lon, P_PN_C_lat = map(p_PN_C_lon, p_PN_C_lat)

map.plot(P_NA_lon, P_NA_lat, marker='D', color='k', markersize=4)

map.plot(P_G_lon, P_G_lat, marker='D', color='#FF0000', markersize=4)
map.plot(P_TT_lon, P_TT_lat, marker='D', color='#8A0808', markersize=4)

map.plot(P_PP_lon, P_PP_lat, marker='D', color='#0B0B3B', markersize=4)
map.plot(P_PP_C_lon, P_PP_C_lat, marker='D', color='#00BFFF', markersize=4)

map.plot(P_PN_lon, P_PN_lat, marker='D', color='#0B3B0B', markersize=4)
map.plot(P_PN_C_lon, P_PN_C_lat, marker='D', color='#00FF40', markersize=4)

map.fillcontinents(color='white')
plt.savefig(u'/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/10_expo_2017/puntos_series.png', bbox_inches='tight', dpi=300)

plt.show()
