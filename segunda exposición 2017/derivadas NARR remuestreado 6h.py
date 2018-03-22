# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 04:54:14 2016

@author: yordan
"""

import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
import pandas as pd

#==============================================================================
'''SELECCIÓN DE SERIES RESOLUCIÓN HORARIA'''
#==============================================================================

ArchivoU = nc.Dataset('/home/yordan/Escritorio/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/NARR/VIENTOS-NARR/ZONAL-REPROYECCION/1979alt.nc')
ArchivoV = nc.Dataset('/home/yordan/Escritorio/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/NARR/VIENTOS-NARR/MERIDIONAL-REPROYECCION/1979alt.nc')

#Variables = [v for v in ArchivoU.variables]
#print Variables

LON = ArchivoV.variables['LON'][:]-360
LAT = ArchivoV.variables['LAT'][:]

posiciones = ['TT','PP','PN']
lon_max = [-96.25, -90.5, -80.75]
lat_max = [16, 11.75, 8.5]
lon_min = [-93.75, -86, -78.75]
lat_min = [12.5, 8.75, 5.5]

datep = pd.date_range('1979-01-01 00:00:00', '2015-12-31 23:00:00', freq='3H')
date = pd.DatetimeIndex(datep)
Serie = pd.DataFrame(index=date, columns=['TT','PP', 'PN']) 

for i, j, k, m, n in zip(lat_max, lat_min, lon_max, lon_min, posiciones):
    for l in range(1979, 2016): # De 0 a 37, porque sólo se va a hacer hasta el 2015
        U = nc.Dataset('/home/yordan/Escritorio/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/NARR/VIENTOS-NARR/ZONAL-REPROYECCION/'+str(l)+'alt.nc')
        V = nc.Dataset('/home/yordan/Escritorio/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/NARR/VIENTOS-NARR/MERIDIONAL-REPROYECCION/'+str(l)+'alt.nc')
        max_lat = np.where(LAT == i)[0][0]
        min_lat = np.where(LAT == j)[0][0]
        max_lon = np.where(LON == k)[0][0]
        min_lon = np.where(LON == m)[0][0]
        
        comp_u = np.zeros(len(U.variables['U']))
        comp_v = np.zeros(len(V.variables['V']))
        
        for x in range(len(comp_u)):
            comp_u[x] = np.mean(U.variables['U'][x, min_lat : max_lat+1, max_lon : min_lon+1])
            comp_v[x] = np.mean(U.variables['U'][x, min_lat : max_lat+1, max_lon : min_lon+1])
        
        spd = np.sqrt(comp_u*comp_u+comp_v*comp_v)
        Serie[n][str(l)+'-01-01 00:00:00':str(l)+'-12-31 23:00:00'] = spd
        
MA = Serie.as_matrix(columns=None)

#==============================================================================
'''AGREGANDO LOS DATOS CADA 6 HORAS'''
#==============================================================================
spd_TT_NARR = np.zeros(len(MA[:,0])/2)
spd_PP_NARR = np.zeros(len(MA[:,0])/2)
spd_PN_NARR = np.zeros(len(MA[:,0])/2)

for i, j in enumerate(range(0, len(MA[:,0]), 2)):
    spd_TT_NARR[i] = np.mean(MA[j,0]+MA[j+1,0])
    spd_PP_NARR[i] = np.mean(MA[j,1]+MA[j+1,1])
    spd_PN_NARR[i] = np.mean(MA[j,2]+MA[j+1,2])
    
#==============================================================================
'''IDENTIFICANDO POSICIONES DONDe ERA SUPERA UMBRALES'''
#==============================================================================

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

########################
"VEOCIDAD EN CADA PIXEL"
########################

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

##################################################################
"IDENTIFICACIÓN DE FECHAS QUE SUPERAN UMBRAL TT-7, PP-6.5, PN-6.5"
##################################################################

Ser_TT = np.zeros(len(spd_TT))
Ser_PP = np.zeros(len(spd_PP))
Ser_PN = np.zeros(len(spd_PN))

###### TT ######
for i in range(len(Ser_TT)):
    if spd_TT[i] > 7:
        Ser_TT[i] = spd_TT[i]

##### PP ######
for i in range(len(Ser_PP)):
    if spd_PP[i] > 6.5:
        Ser_PP[i] = spd_PP[i]

#### PN ######
for i in range(len(Ser_PN)):
    if spd_PN[i] > 6.5:
        Ser_PN[i] = spd_PN[i]

###################################
"ÍNDICES EN LOS QUE HAY EVENTOS TT"
###################################
i = 0
evn_TT = []

while i<(len(Ser_TT)-1):    
    if Ser_TT[i] != 0: # Primero se verifica que la posición i tenga un valor
        j = i
        l = 3 # Esto es para poder entrar al while
        while l <= 4:        
            if j < (len(Ser_TT)-1): # éste condicional se hace ya que en el caso de que j sea una unidad menor a la longitud del vector ser, la siguiente línea arrojará un error
                while Ser_TT[j+1] != 0: # Se hace éste while para saber hasta qué posición llega el evento 
                    j = j+1
                    if j == (len(Ser_TT)-1): # Se debe parar en el momento en que j alcance una unidad menos a la longitud de ser, ya que de lo contrario, cuando se evalúe la condición del bucle se tendría un error
                        break
                
                if j < (len(Ser_TT)-5): # Esta línea es debida a la siguiente, para que no se genere un error
                    if np.any(Ser_TT[j+1:j+5] != np.zeros(4)) == True: # Se debe revisar que las siguientes cuatro posiciones a la posición j no tengan un valor, ya que de lo contrario, esos valores harán parte del evento al que pertenece el valor de la posición j
                        for k in range(j+1, j+5): # para hallar en qué posición de las cuatro que se juzgan se encuentra la posición con valor distinto de cero 
                            if Ser_TT[k] != 0:
                                l = 2 # Para que el while se vuelva a repetir
                                j = k
                                break
                            else:
                                l = 5 # Para que el while no se vuelva a repetir
                    else:
                        l = 5 # Para que el while no se vuelva a repetir
                
                elif j < (len(Ser_TT)-4): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-5
                    if np.any(Ser_TT[j+1:j+4] != np.zeros(3)) == True:
                        for k in range(j+1, j+4):
                            if Ser_TT[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                        
                
                elif j < (len(Ser_TT)-3): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-4
                    if np.any(Ser_TT[j+1:j+3] != np.zeros(2)) == True:
                        for k in range(j+1, j+3):
                            if Ser_TT[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5
                
                elif j < (len(Ser_TT)-2):# Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-3
                    if np.any(Ser_TT[j+1:j+2] != 0) == True:
                        for k in range(j+1, j+2):
                            if Ser_TT[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                
                else:
                    l = 5
            else:
                l = 5
        evn_TT.append((i,j))       
        i = j+4 # ésta sería la nueva posición donde comienza todo el bucle     
    else:
        i = i+1 # Siguiente posicion a examinar
EVN_TT = np.array((evn_TT))

###################################
"ÍNDICES EN LOS QUE HAY EVENTOS PP"
###################################
i = 0
evn_PP = []

while i<(len(Ser_PP)-1):    
    if Ser_PP[i] != 0: # Primero se verifica que la posición i tenga un valor
        j = i
        l = 3
        while l <= 4:        
            if j < (len(Ser_PP)-1): # éste condicional se hace ya que en el caso de que j sea una unidad menor a la longitud del vector ser, la siguiente línea arrojará un error
                while Ser_PP[j+1] != 0: # Se hace éste while para saber hasta qué posición llega el evento 
                    j = j+1
                    if j == (len(Ser_PP)-1): # Se debe parar en el momento en que j alcance una unidad menos a la longitud de ser, ya que de lo contrario, cuando se evalúe la condición del bucle se tendría un error
                        break
                
                if j < (len(Ser_PP)-5): # Esta línea es debida a la siguiente, para que no se genere un error
                    if np.any(Ser_PP[j+1:j+5] != np.zeros(4)) == True: # Se debe revisar que las siguientes cuatro posiciones a la posición j no tengan un valor, ya que de lo contrario, esos valores harán parte del evento al que pertenece el valor de la posición j
                        for k in range(j+1, j+5): # para hallar en qué posición de las cuatro que se juzgan se encuentra la posición con valor distinto de cero 
                            if Ser_PP[k] != 0:
                                l = 2 # Para que el while se vuelva a repetir
                                j = k
                                break
                            else:
                                l = 5 # Para que el while no se vuelva a repetir
                    else:
                        l = 5 # Para que el while no se vuelva a repetir
                elif j < (len(Ser_PP)-4): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-5
                    if np.any(Ser_PP[j+1:j+4] != np.zeros(3)) == True:
                        for k in range(j+1, j+4):
                            if Ser_PP[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                        
                elif j < (len(Ser_PP)-3): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-4
                    if np.any(Ser_PP[j+1:j+3] != np.zeros(2)) == True:
                        for k in range(j+1, j+3):
                            if Ser_PP[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5
                elif j < (len(Ser_PP)-2):# Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-3
                    if np.any(Ser_PP[j+1:j+2] != 0) == True:
                        for k in range(j+1, j+2):
                            if Ser_PP[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                
                else:
                    l = 5
            else:
                l = 5
        evn_PP.append((i,j))       
        i = j+4 # ésta sería la nueva posición donde comienza todo el bucle     
    else:
        i = i+1 # Siguiente posicion a examinar
EVN_PP = np.array((evn_PP))

###################################
"ÍNDICES EN LOS QUE HAY EVENTOS PN"
###################################
i = 0
evn_PN = []

while i<(len(Ser_PN)-1):    
    if Ser_PN[i] != 0: # Primero se verifica que la posición i tenga un valor
        j = i
        l = 3
        while l <= 4:        
            if j < (len(Ser_PN)-1): # éste condicional se hace ya que en el caso de que j sea una unidad menor a la longitud del vector ser, la siguiente línea arrojará un error
                while Ser_PN[j+1] != 0: # Se hace éste while para saber hasta qué posición llega el evento 
                    j = j+1
                    if j == (len(Ser_PN)-1): # Se debe parar en el momento en que j alcance una unidad menos a la longitud de ser, ya que de lo contrario, cuando se evalúe la condición del bucle se tendría un error
                        break
                
                if j < (len(Ser_PN)-5): # Esta línea es debida a la siguiente, para que no se genere un error
                    if np.any(Ser_PN[j+1:j+5] != np.zeros(4)) == True: # Se debe revisar que las siguientes cuatro posiciones a la posición j no tengan un valor, ya que de lo contrario, esos valores harán parte del evento al que pertenece el valor de la posición j
                        for k in range(j+1, j+5): # para hallar en qué posición de las cuatro que se juzgan se encuentra la posición con valor distinto de cero 
                            if Ser_PN[k] != 0:
                                l = 2 # Para que el while se vuelva a repetir
                                j = k
                                break
                            else:
                                l = 5 # Para que el while no se vuelva a repetir
                    else:
                        l = 5 # Para que el while no se vuelva a repetir
                elif j < (len(Ser_PN)-4): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-5
                    if np.any(Ser_PN[j+1:j+4] != np.zeros(3)) == True:
                        for k in range(j+1, j+4):
                            if Ser_PN[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                        
                elif j < (len(Ser_PN)-3): # Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-4
                    if np.any(Ser_PN[j+1:j+3] != np.zeros(2)) == True:
                        for k in range(j+1, j+3):
                            if Ser_PN[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5
                elif j < (len(Ser_PN)-2):# Éste condicional tiene lugar cuando en la identificación de la posición j, ésta toma el valor de len(ser)-3
                    if np.any(Ser_PN[j+1:j+2] != 0) == True:
                        for k in range(j+1, j+2):
                            if Ser_PN[k] != 0:
                                l = 2
                                j = k
                                break
                            else:
                                l = 5
                    else:
                        l = 5                
                else:
                    l = 5
            else:
                l = 5
        evn_PN.append((i,j))       
        i = j+4 # ésta sería la nueva posición donde comienza todo el bucle     
    else:
        i = i+1 # Siguiente posicion a examinar
EVN_PN = np.array((evn_PN))


################ 
"DERIVADA EN TT"
################

for m in range(19):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    for i in range(len(EVN_TT[:,0])):
        if EVN_TT[i,0] != EVN_TT[i,1]:
            c = EVN_TT[i,0]+np.where(spd_TT_NARR[EVN_TT[i,0]:EVN_TT[i,1]+1] == np.max(spd_TT_NARR[EVN_TT[i,0]:EVN_TT[i,1]+1]))
            cc = c[0,0]
            if cc <= len(spd_TT_NARR)-7:        
                if m <= spd_TT[cc-6] < m+1:
                    der = np.zeros(len(spd_TT_NARR[cc-6:cc+7]))                
                    for j in range(len(der)):
                        der[j] = spd_TT_NARR[cc-6+j+1]-spd_TT_NARR[cc-6+j]
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12) ,der, linewidth=0.7, color='k', alpha = 0.7)
        else:
            if EVN_TT[i,1] <= len(spd_TT_NARR)-7:        
                if m <= spd_TT[EVN_TT[i,1]-6]< m+1:
                    der = np.zeros(len(spd_TT_NARR[cc-6:cc+7]))
                    for j in range(len(der)):
                        der[j] = spd_TT_NARR[EVN_TT[i,1]-6+j+1]-spd_TT_NARR[EVN_TT[i,1]-6+j] 
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12), der, linewidth=0.7, color='k', alpha = 0.7)
    my_xticks = ['-36:00','-30:00','-24:00', '-18:00', '-12:00', '-06:00', '00:00', '06:00', '12:00', '18:00', '24:00', '30:00', '36:00']
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),my_xticks)
    ax1.grid(True)
    ax1.set_xlabel('$Time$ $(h)$', size='15')
    ax1.set_ylabel('$\Delta$ $Speed$ $(m/s)$', size='15')
    ax1.set_title('Derived from Wind Speed, '+str(float(m))+' - '+str(float(m+1))+' (TT)', size='14')
    ax1.legend(loc='best')
    ax1.set_ylim(-15, 15)
    plt.savefig('/home/yordan/YORDAN/UNAL/TESIS MAESTRÍA/segunda exposición 2017/TT/'+str(m)+'-'+str(m+1)+'.png',dpi=100,bbox_inches='tight')

################ 
"DERIVADA EN PP"
################

for m in range(14):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    for i in range(len(EVN_PP[:,0])):
        if EVN_PP[i,0] != EVN_PP[i,1]:
            c = EVN_PP[i,0]+np.where(spd_PP_NARR[EVN_PP[i,0]:EVN_PP[i,1]+1] == np.max(spd_PP_NARR[EVN_PP[i,0]:EVN_PP[i,1]+1]))
            cc = c[0,0]
            if cc <= len(spd_PP_NARR)-7:        
                if m <= spd_PP[cc-6] < m+1:
                    der = np.zeros(len(spd_PP_NARR[cc-6:cc+7]))                
                    for j in range(len(der)):
                        der[j] = spd_PP_NARR[cc-6+j+1]-spd_PP_NARR[cc-6+j]
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12) ,der, linewidth=0.7, color='k', alpha = 0.7)
        else:
            if EVN_PP[i,1] <= len(spd_PP_NARR)-7:        
                if m <= spd_PP[EVN_PP[i,1]-6]< m+1:
                    der = np.zeros(len(spd_PP_NARR[cc-6:cc+7]))
                    for j in range(len(der)):
                        der[j] = spd_PP_NARR[EVN_PP[i,1]-6+j+1]-spd_PP_NARR[EVN_PP[i,1]-6+j] 
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12), der, linewidth=0.7, color='k', alpha = 0.7)
    my_xticks = ['-36:00','-30:00','-24:00', '-18:00', '-12:00', '-06:00', '00:00', '06:00', '12:00', '18:00', '24:00', '30:00', '36:00']
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),my_xticks)
    ax1.grid(True)
    ax1.set_xlabel('$Time$ $(h)$', size='15')
    ax1.set_ylabel('$\Delta$ $Speed$ $(m/s)$', size='15')
    ax1.set_title('Derived from Wind Speed, '+str(float(m))+' - '+str(float(m+1))+' (PP)', size='14')
    ax1.legend(loc='best')
    ax1.set_ylim(-15, 15)
    plt.savefig('/home/yordan/YORDAN/UNAL/TESIS MAESTRÍA/segunda exposición 2017/PP/'+str(m)+'-'+str(m+1)+'.png',dpi=100,bbox_inches='tight')


################ 
"DERIVADA EN PN"
################

spd_PN

for m in range(13):
    fig = plt.figure(figsize=(8,6))
    ax1 = fig.add_subplot(111)
    for i in range(len(EVN_PN[:,0])):
        if EVN_PN[i,0] != EVN_PN[i,1]:
            c = EVN_PN[i,0]+np.where(spd_PN_NARR[EVN_PN[i,0]:EVN_PN[i,1]+1] == np.max(spd_PN_NARR[EVN_PN[i,0]:EVN_PN[i,1]+1]))
            cc = c[0,0]
            if cc <= len(spd_PN_NARR)-7:        
                if m <= spd_PN[cc-6] < m+1:
                    der = np.zeros(len(spd_PN_NARR[cc-6:cc+7]))                
                    for j in range(len(der)):
                        der[j] = spd_PN_NARR[cc-6+j+1]-spd_PN_NARR[cc-6+j]
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12) ,der, linewidth=0.7, color='k', alpha = 0.7)
        else:
            if EVN_PN[i,1] <= len(spd_PN_NARR)-7:        
                if m <= spd_PN[EVN_PN[i,1]-6]< m+1:
                    der = np.zeros(len(spd_PN_NARR[cc-6:cc+7]))
                    for j in range(len(der)):
                        der[j] = spd_PN_NARR[EVN_PN[i,1]-6+j+1]-spd_PN_NARR[EVN_PN[i,1]-6+j] 
                    ax1.plot((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12), der, linewidth=0.7, color='k', alpha = 0.7)
    my_xticks = ['-36:00','-30:00','-24:00', '-18:00', '-12:00', '-06:00', '00:00', '06:00', '12:00', '18:00', '24:00', '30:00', '36:00']
    plt.xticks((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 , 11, 12),my_xticks)
    ax1.grid(True)
    ax1.set_xlabel('$Time$ $(h)$', size='15')
    ax1.set_ylabel('$\Delta$ $Speed$ $(m/s)$', size='15')
    ax1.set_title('Derived from Wind Speed, '+str(float(m))+' - '+str(float(m+1))+' (PN)', size='14')
    ax1.legend(loc='best')
    ax1.set_ylim(-10, 12)
    plt.savefig('/home/yordan/YORDAN/UNAL/TESIS MAESTRÍA/segunda exposición 2017/PN/'+str(m)+'-'+str(m+1)+'.png',dpi=100,bbox_inches='tight')




