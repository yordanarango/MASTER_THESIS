# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 13:34:25 2018

@author: yordan
"""
import numpy as np
import matplotlib.pyplot as plt

"Datos de Log-Likelihood"
LogTT_Beta  = np.array([20388.06, 25308.08, 28838.65, 31152.2 , 32744.57])
LogTT_Gamma = np.array([20485.19, 25747.85, 29137.71, 31442.84, 32942.6])
LogTT_LNorm = np.array([18634.74, 24843.76, 28218.6 , 30639.44, 32146.4])
LogTT_Logis = np.array([20249.55])

# LogPP_Beta  = np.array([])
# LogPP_Gamma = np.array([])
# LogPP_LNorm = np.array([])
# LogPP_Logis = np.array([])

# LogPN_Beta  = np.array([])
# LogPN_Gamma = np.array([])
# LogPN_LNorm = np.array([])
# LogPN_Logis = np.array([])


"Datos de AIC"
AIC_TT_Beta  = - 2 * LogTT_Beta  + 2 * (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1)
AIC_TT_Gamma = - 2 * LogTT_Gamma + 2 * (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1)
AIC_TT_LNorm = - 2 * LogTT_LNorm + 2 * (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1)
AIC_TT_Logis = - 2 * LogTT_Logis + 2 * (        2**2      +         2      * 2 - 1)

# AIC_PP_Beta  = - 2 * LogPP_Beta  + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
# AIC_PP_Gamma = - 2 * LogPP_Gamma + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
# AIC_PP_LNorm = - 2 * LogPP_LNorm + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
# AIC_PP_Logis = - 2 * LogPP_Logis + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)

# AIC_PN_Beta  = - 2 * LogPN_Beta  + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
# AIC_PN_Gamma = - 2 * LogPN_Gamma + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
# AIC_PN_LNorm = - 2 * LogPN_LNorm + 2 * (np.arange(2,5) **2 + np.arange(2,5)  * 2 - 1)
# AIC_PN_Logis = - 2 * LogPN_Logis + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)


"Datos de BIC"
BIC_TT_Beta  = - 2 * LogTT_Beta  + (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1) * np.log(27552)
BIC_TT_Gamma = - 2 * LogTT_Gamma + (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1) * np.log(27552)
BIC_TT_LNorm = - 2 * LogTT_LNorm + (np.arange(2,7)**2 + np.arange(2,7) * 2 - 1) * np.log(27552)
BIC_TT_Logis = - 2 * LogTT_Logis + (      2**2        +        2       * 2 - 1) * np.log(27552)

# BIC_PP_Beta  = - 2 * LogPP_Beta  + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
# BIC_PP_Gamma = - 2 * LogPP_Gamma + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
# BIC_PP_LNorm = - 2 * LogPP_LNorm + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
# BIC_PP_Logis = - 2 * LogPP_Logis + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)

# BIC_PN_Beta  = - 2 * LogPN_Beta  + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
# BIC_PN_Gamma = - 2 * LogPN_Gamma + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
# BIC_PN_LNorm = - 2 * LogPN_LNorm + (np.arange(2,5) **2 + np.arange(2,5)  * 2 - 1) * np.log(6888)
# BIC_PN_Logis = - 2 * LogPN_Logis + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)


"plot TT"

'AIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,7), AIC_TT_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,7), AIC_TT_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,7), AIC_TT_LNorm, c='b', label='LogNormal' )
plt.plot(       2,       AIC_TT_Logis, c='g', label='Logis'     )

plt.xlim(1, 7)

plt.xlabel('States', fontsize=14)
plt.ylabel('AIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("AIC - TT's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/validation/TT_AIC.png', bbox_inches='tight', dpi=300)
plt.close('all')



'BIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,7), BIC_TT_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,7), BIC_TT_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,7), BIC_TT_LNorm, c='b', label='LogNormal' )
plt.plot(       2      , BIC_TT_Logis, c='g', label='Logis'     )

plt.xlim(1, 7)

plt.xlabel('States', fontsize=14)
plt.ylabel('BIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("BIC - TT's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/validation/TT_BIC.png', bbox_inches='tight', dpi=300)
plt.close('all')

















"plot PP"

'AIC'
# plt.figure(figsize=(10,5))

# plt.plot(np.arange(2,11), AIC_PP_Beta,  c='k', label='Beta'      )
# plt.plot(np.arange(2,11), AIC_PP_Gamma, c='r', label='Gamma'     )
# plt.plot(np.arange(2,11), AIC_PP_LNorm, c='b', label='LogNormal' )
# plt.plot(np.arange(2,11), AIC_PP_Logis, c='g', label='Logis'     )

# plt.xlim(1, 11)

# plt.xlabel('States', fontsize=14)
# plt.ylabel('AIC', fontsize=14)

# plt.xticks(size=13)
# plt.yticks(size=13)

# plt.title("AIC - PP's HMM", fontsize=14)
# plt.grid('True')
# plt.legend(loc='right center')

# plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PP_AIC.png', bbox_inches='tight', dpi=300)
# plt.close('all')



'BIC'
# plt.figure(figsize=(10,5))

# plt.plot(np.arange(2,11), BIC_PP_Beta,  c='k', label='Beta'      )
# plt.plot(np.arange(2,11), BIC_PP_Gamma, c='r', label='Gamma'     )
# plt.plot(np.arange(2,11), BIC_PP_LNorm, c='b', label='LogNormal' )
# plt.plot(np.arange(2,11), BIC_PP_Logis, c='g', label='Logis'     )

# plt.xlim(1, 11)

# plt.xlabel('States', fontsize=14)
# plt.ylabel('BIC', fontsize=14)

# plt.xticks(size=13)
# plt.yticks(size=13)

# plt.title("BIC - PP's HMM", fontsize=14)
# plt.grid('True')
# plt.legend(loc='right center')

# plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PP_BIC.png', bbox_inches='tight', dpi=300)
# plt.close('all')














"plot PN"

'AIC'
# plt.figure(figsize=(10,5))

# plt.plot(np.arange(2,11), AIC_PN_Beta,  c='k', label='Beta'      )
# plt.plot(np.arange(2,11), AIC_PN_Gamma, c='r', label='Gamma'     )
# plt.plot(np.arange(2,5),  AIC_PN_LNorm, c='b', label='LogNormal' )
# plt.plot(np.arange(2,11), AIC_PN_Logis, c='g', label='Logis'     )

# plt.xlim(1, 11)

# plt.xlabel('States', fontsize=14)
# plt.ylabel('AIC', fontsize=14)

# plt.xticks(size=13)
# plt.yticks(size=13)

# plt.title("AIC - PN's HMM", fontsize=14)
# plt.grid('True')
# plt.legend(loc='right center')

# plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PN_AIC.png', bbox_inches='tight', dpi=300)
# plt.close('all')



'BIC'
# plt.figure(figsize=(10,5))

# plt.plot(np.arange(2,11), BIC_PN_Beta,  c='k', label='Beta'      )
# plt.plot(np.arange(2,11), BIC_PN_Gamma, c='r', label='Gamma'     )
# plt.plot(np.arange(2,5),  BIC_PN_LNorm, c='b', label='LogNormal' )
# plt.plot(np.arange(2,11), BIC_PN_Logis, c='g', label='Logis'     )

# plt.xlim(1, 11)

# plt.xlabel('States', fontsize=14)
# plt.ylabel('BIC', fontsize=14)

# plt.xticks(size=13)
# plt.yticks(size=13)

# plt.title("BIC - PN's HMM", fontsize=14)
# plt.grid('True')
# plt.legend(loc='right center')

# plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PN_BIC.png', bbox_inches='tight', dpi=300)
# plt.close('all')



