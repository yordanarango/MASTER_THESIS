# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 06:51:00 2017

@author: yordan
"""


import numpy as np
from statsmodels.formula.api import ols
import pandas as pd
import xlrd
import pylab as plt
import netCDF4 as nc

Archivo = nc.Dataset('/home/yordan/Escritorio/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')

######################################## 
"ZONAS DE ALTA INFLUENCIA EN TT, PP, PN"
########################################

u_TT = Archivo.variables['u10'][:54056, 4:19, 7:18] # serie desde 1979/01/01 hasta 2015/12/31
v_TT = Archivo.variables['v10'][:54056, 4:19, 7:18]
u_PP = Archivo.variables['u10'][:54056, 21:34, 30:49]
v_PP = Archivo.variables['v10'][:54056, 21:34, 30:49]
u_PN = Archivo.variables['u10'][:54056, 34:47, 69:78]
v_PN = Archivo.variables['v10'][:54056, 34:47, 69:78]

#########################
"VELOCIDAD EN CADA PIXEL"
#########################

velTT = np.sqrt(u_TT*u_TT+v_TT*v_TT) 
velPP = np.sqrt(u_PP*u_PP+v_PP*v_PP)
velPN = np.sqrt(u_PN*u_PN+v_PN*v_PN)

###################### 
"COMPOSITE TT, PP, PN"
######################

spd_TT = np.zeros(54056)
spd_PP = np.zeros(54056)
spd_PN = np.zeros(54056)

for i in range(54056):
    spd_TT[i] = np.mean(velTT[i,:,:])
    spd_PP[i] = np.mean(velPP[i,:,:])
    spd_PN[i] = np.mean(velPN[i,:,:])
    
#==============================================================================
'''ESCRIBIENDO NETCDF'''
#==============================================================================

N = spd_TT.size
t0 = 1979
dt = 1/1460.
time = np.arange(0, N) * dt + t0 



data = nc.Dataset('/home/yordan/Escritorio/vientos.nc', 'w', format='NETCDF4_CLASSIC')
#dato = data.createGroup('PRECIPITACION_COLOMBIA')
SPD_TT = data.createDimension('SPD_TT', len(spd_TT))
SPD_PP = data.createDimension('SPD_PP', len(spd_PP))
SPD_PN = data.createDimension('SPD_PN', len(spd_PN))
tiempo = data.createDimension('tiempo', len(time))

vel_TT = data.createVariable('SPD_TT','f4',('SPD_TT'))
vel_PP = data.createVariable('SPD_PP','f4',('SPD_PP'))
vel_PN = data.createVariable('SPD_PN','f4',('SPD_PN'))
TIME   = data.createVariable('tiempo','f4',('tiempo'))

vel_TT[:] = spd_TT
vel_PP[:] = spd_PP
vel_PN[:] = spd_PN
TIME[:] = time

data.close()




Archivo = nc.Dataset('/home/yordan/Escritorio/vientos.nc', 'r+')

PP = Archivo.variables['SPD_PP'][:]
