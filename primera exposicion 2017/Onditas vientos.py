# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 07:37:16 2017

@author: yordan
"""

import numpy as np
from statsmodels.formula.api import ols
import pandas as pd
import xlrd
import pylab as plt
import wavelet
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

fechas = pd.date_range('1979/01/01', '2015/12/31 23:00:00', freq='6H')

SPD_TT = pd.Series(data=spd_TT, index=fechas)
SPD_PP = pd.Series(data=spd_PP, index=fechas)
SPD_PN = pd.Series(data=spd_PN, index=fechas)


##############
"CICLO DIURNO"
##############

#serie_TT = spd_TT 
#media_TT = np.zeros((12,4)) # Las filas representan los meses, y las columnas los cuatro horarios dentro de un día
#fechas = pd.date_range('1979/01/01', periods=54056, freq='6H') # serie con fechas cada 6 horas, desde 1979/01/01 hasta 2015/12/31
#
#for i in range(1,13): # Se recorre cada mes
#    for j in range(0, 19, 6): # Se recorre cada horario
#        selec = [] # Cada horario de cada mes, se debe vovler a crear la lista, que es donde se guardarán los datos de horario de cada mes
#        for k in range(len(serie_TT)): # Se recorre toda la serie
#            if fechas[k].month == i: # Se verifica que la fecha que corresponde al valor de la serie, cumpla la condición
#                if fechas[k].hour == j: # Se verifica que el horario que corresponde al valor de la serie, cumpla la condición
#                    selec.append(serie_TT[k]) # Si cumple, se adiciona a la lista
#        media_TT[i-1, j/6] = np.mean(selec) # Se guarda wl valor de la media
#
#
#
#
#serie_PP = spd_PP 
#media_PP = np.zeros((12,4)) # Las filas representan los meses, y las columnas los cuatro horarios dentro de un día
#fechas = pd.date_range('1979/01/01', periods=54056, freq='6H') # serie con fechas cada 6 horas, desde 1979/01/01 hasta 2015/12/31
#
#for i in range(1,13): # Se recorre cada mes
#    for j in range(0, 19, 6): # Se recorre cada horario
#        selec = [] # Cada horario de cada mes, se debe vovler a crear la lista, que es donde se guardarán los datos de horario de cada mes
#        for k in range(len(serie_PP)): # Se recorre toda la serie
#            if fechas[k].month == i: # Se verifica que la fecha que corresponde al valor de la serie, cumpla la condición
#                if fechas[k].hour == j: # Se verifica que el horario que corresponde al valor de la serie, cumpla la condición
#                    selec.append(serie_PP[k]) # Si cumple, se adiciona a la lista
#        media_PP[i-1, j/6] = np.mean(selec) # Se guarda wl valor de la media
#
#
#
#
#serie_PN = spd_PN 
#media_PN = np.zeros((12,4)) # Las filas representan los meses, y las columnas los cuatro horarios dentro de un día
#fechas = pd.date_range('1979/01/01', periods=54056, freq='6H') # serie con fechas cada 6 horas, desde 1979/01/01 hasta 2015/12/31
#
#for i in range(1,13): # Se recorre cada mes
#    for j in range(0, 19, 6): # Se recorre cada horario
#        selec = [] # Cada horario de cada mes, se debe volver a crear la lista, que es donde se guardarán los datos de horario de cada mes
#        for k in range(len(serie_PN)): # Se recorre toda la serie
#            if fechas[k].month == i: # Se verifica que la fecha que corresponde al valor de la serie, cumpla la condición
#                if fechas[k].hour == j: # Se verifica que el horario que corresponde al valor de la serie, cumpla la condición
#                    selec.append(serie_PN[k]) # Si cumple, se adiciona a la lista
#        media_PN[i-1, j/6] = np.mean(selec) # Se guarda wl valor de la media

####################
"ANOMALÍAS HORARIAS"
####################        
#ano_TT = np.zeros(len(serie_TT))
#for i in range(len(serie_TT)):
#    ano_TT[i] = serie_TT[i]-media_TT[fechas[i].month-1, fechas[i].hour/6]
#
#
#ano_PP = np.zeros(len(serie_PP))
#for i in range(len(serie_PP)):
#    ano_PP[i] = serie_PP[i]-media_PP[fechas[i].month-1, fechas[i].hour/6]
#
#
#ano_PN = np.zeros(len(serie_PN))
#for i in range(len(serie_PN)):
#    ano_PN[i] = serie_PN[i]-media_PN[fechas[i].month-1, fechas[i].hour/6]

#==============================================================================
'''REMOVIENDO CICLO ANUAL'''
#==============================================================================

def anomalias(serie): #ingresa serie de pandas
    seriestand=[];indice=[]
    for i in np.arange(1,13): #arange para identicon los meses para identificarlos
        pos=np.where(serie.index.month==i)[0] #encuentra la posición donde están los meses
        media=(np.nanmean(serie[pos], axis =0)) #saca las medias de todos los meses
        seriestand.extend(serie[pos]-media) #estandariza quitando el ciclo anual
        indice.extend(serie.index[pos]) #crea los índices para los nuevos datos estandarizados
    return pd.Series(seriestand,index=indice).sort_index()

ano_TT = anomalias(SPD_TT)
ano_PP = anomalias(SPD_PP)
ano_PN = anomalias(SPD_PN)




label = 'Wind PN'

t0 = 1979

dt = 1/1460. #Tasa de muestreo en años
units = 'm/s'
var = ano_PP

#==============================================================================
'''Transformada de onditas, inversa de la tranformada de onditas y descomposición'''
#==============================================================================
slevel = 0.95                        # Significance level
std = var.std()                      # Standard deviation
std2 = std ** 2                      # Variance
#var = (var - var.mean()) / std       # Calculating anomaly and normalizing

N = var.size                         # Number of measurements
time = np.arange(0, N) * dt + t0  # Time array in years

dj = 0.25                       # Four sub-octaves per octaves (lo puse igual que el dt, este es alfa de la tesis de carlos)
s0 = 2*dt #2 * dt                      # Starting scale, here 6 months
J = (np.log2(N*dt/s0))/dj #7 / dj                      # Seven powers of two with dj sub-octaves
#alpha = 0.0                          # Lag-1 autocorrelation for white noise
#alpha, _, _ = wavelet.ar1(var)
alpha = np.corrcoef(var[0:-1], var[1:])[0,1]
mother = wavelet.Morlet(6.)          # Morlet mother wavelet with wavenumber=6

# The following routines perform the wavelet transform and siginificance
# analysis for the chosen data set.
wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(var, dt, dj, s0, J,
                                                      mother)
iwave = wavelet.icwt(wave, scales, dt, dj, mother)
power = (abs(wave)) ** 2             # Normalized wavelet power spectrum
fft_power = std2 * abs(fft) ** 2     # FFT power spectrum
period = 1. / freqs

signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, alpha,
                        significance_level=slevel, wavelet=mother)
sig95 = np.ones([1, N]) * signif[:, None]
sig95 = power / sig95                # Where ratio > 1, power is significant



#the wavelet power spectrum
fig = plt.figure(figsize=(10,7))
# The following routines plot the results in four different subplots containing
# the original series anomaly, the wavelet power spectrum, the global wavelet
# and Fourier spectra and finally the range averaged wavelet spectrum. In all
# sub-plots the significance levels are either included as dotted lines or as
# filled contour lines.
#bx = plt.axes([0.1, 0.37, 0.65, 0.28])
bx = plt.axes([0.2, 0.37, 0.65, 0.28])
W = bx.contourf(time, np.log2(period), np.log2(power), np.arange(0, 10), extend='both')
#bx.contour(time, np.log2(period), sig95, [-99, 1], colors='k',linewidths=1.)
bx.fill(np.concatenate([time,time[:1]-dt, time[-1:]+dt, time[-1:]+dt,
        time[:1]-dt, time[:1]-dt]), np.log2(np.concatenate([[1e-9], coi,
        [1e-9], period[-1:], period[-1:], [1e-9]])), 'k', alpha=0.3,
        hatch='x')

'''LO INTRODUZCO YO'''
ce = plt.colorbar(W, drawedges = 'True',format='%.1f')
ce.set_label('m/s')





bx.set_title('%s Wavelet Power Spectrum (%s)' % (label, mother.name))
bx.set_ylabel('Period (Months)')
Yticks = 2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
Yticks_label = (2 ** np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max()))))*12
bx.set_yticks(np.log2(Yticks))
bx.set_yticklabels(Yticks_label)
bx.set_ylim(np.log2([period.min(), period.max()]))
bx.invert_yaxis()


# That's all folks!
