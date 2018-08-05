# -*- coding: utf-8 -*-
"""
Created on Fri Jun 06 13:34:25 2018

@author: yordan
"""
import numpy as np
import matplotlib.pyplot as plt

"Datos de Log-Likelihood"
LogTT_Beta  = np.array([3802.653, 4133.956, 4376.062, 4485.845, 4603.139, 4637.432, 4690.413, 4695.168, 4705.981])
LogTT_Gamma = np.array([3812.001, 4221.883, 4468.213, 4559.243, 4610.768, 4659.525, 4703.974, 4742.67,  4765.993])
LogTT_LNorm = np.array([3467.332, 3962.247, 4225.286, 4474.548, 4567.293, 4611.315])
LogTT_Logis = np.array([2505.539, 2505.539, 4356.151, 4471.464, 4540.98,  4600.592, 2505.539, 2505.539, 2505.539])

LogPP_Beta  = np.array([4195.801, 4747.334, 5070.126, 5297.89,  5510.717, 5547.599, 5662.284, 5808.702, 5828.683])
LogPP_Gamma = np.array([4247.652, 4857.49,  5171.026, 5263.384, 5511.216, 5649.844, 5736.287, 5792.624, 5862.072])
LogPP_LNorm = np.array([3543.301, 4388.89,  5001.113, 5081.738, 5365.551, 5543.139, 5683.728, 5720.609, 5823.071])
LogPP_Logis = np.array([4377.072, 4996.581, 5313.637, 5405.162, 5566.056, 5668.384, 5759.008, 5804.753, 5849.471])

LogPN_Beta  = np.array([3867.3,   4278.322, 4611.421, 4777.576, 4906.691, 4983.04,  5040.767, 5048.658, 5099.441])
LogPN_Gamma = np.array([3913.409, 4429.639, 4717.573, 4848.313, 4925.071, 4983.48,  5006.38,  5061.863, 5097.185])
LogPN_LNorm = np.array([3192.817, 4150.291, 4486.881])
LogPN_Logis = np.array([4113.518, 4504.714, 4708.468, 4855.74,  4928.108, 4983.826, 5024.443, 5037.331, 5104.929])


"Datos de AIC"
AIC_TT_Beta  = - 2 * LogTT_Beta  + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_TT_Gamma = - 2 * LogTT_Gamma + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_TT_LNorm = - 2 * LogTT_LNorm + 2 * (np.arange(2,8) **2 + np.arange(2,8)  * 2 - 1)
AIC_TT_Logis = - 2 * LogTT_Logis + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)

AIC_PP_Beta  = - 2 * LogPP_Beta  + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_PP_Gamma = - 2 * LogPP_Gamma + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_PP_LNorm = - 2 * LogPP_LNorm + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_PP_Logis = - 2 * LogPP_Logis + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)

AIC_PN_Beta  = - 2 * LogPN_Beta  + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_PN_Gamma = - 2 * LogPN_Gamma + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)
AIC_PN_LNorm = - 2 * LogPN_LNorm + 2 * (np.arange(2,5) **2 + np.arange(2,5)  * 2 - 1)
AIC_PN_Logis = - 2 * LogPN_Logis + 2 * (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1)


"Datos de BIC"
BIC_TT_Beta  = - 2 * LogTT_Beta  + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_TT_Gamma = - 2 * LogTT_Gamma + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_TT_LNorm = - 2 * LogTT_LNorm + (np.arange(2,8) **2 + np.arange(2,8)  * 2 - 1) * np.log(6888)
BIC_TT_Logis = - 2 * LogTT_Logis + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)

BIC_PP_Beta  = - 2 * LogPP_Beta  + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_PP_Gamma = - 2 * LogPP_Gamma + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_PP_LNorm = - 2 * LogPP_LNorm + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_PP_Logis = - 2 * LogPP_Logis + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)

BIC_PN_Beta  = - 2 * LogPN_Beta  + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_PN_Gamma = - 2 * LogPN_Gamma + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)
BIC_PN_LNorm = - 2 * LogPN_LNorm + (np.arange(2,5) **2 + np.arange(2,5)  * 2 - 1) * np.log(6888)
BIC_PN_Logis = - 2 * LogPN_Logis + (np.arange(2,11)**2 + np.arange(2,11) * 2 - 1) * np.log(6888)


"plot TT"

'AIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), AIC_TT_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), AIC_TT_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,8),  AIC_TT_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), AIC_TT_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('AIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("AIC - TT's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/TT_AIC.png', bbox_inches='tight', dpi=300)
plt.close('all')



'BIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), BIC_TT_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), BIC_TT_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,8),  BIC_TT_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), BIC_TT_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('BIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("BIC - TT's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/TT_BIC.png', bbox_inches='tight', dpi=300)
plt.close('all')

















"plot PP"

'AIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), AIC_PP_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), AIC_PP_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,11), AIC_PP_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), AIC_PP_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('AIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("AIC - PP's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PP_AIC.png', bbox_inches='tight', dpi=300)
plt.close('all')



'BIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), BIC_PP_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), BIC_PP_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,11), BIC_PP_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), BIC_PP_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('BIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("BIC - PP's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PP_BIC.png', bbox_inches='tight', dpi=300)
plt.close('all')














"plot PN"

'AIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), AIC_PN_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), AIC_PN_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,5),  AIC_PN_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), AIC_PN_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('AIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("AIC - PN's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PN_AIC.png', bbox_inches='tight', dpi=300)
plt.close('all')



'BIC'
plt.figure(figsize=(10,5))

plt.plot(np.arange(2,11), BIC_PN_Beta,  c='k', label='Beta'      )
plt.plot(np.arange(2,11), BIC_PN_Gamma, c='r', label='Gamma'     )
plt.plot(np.arange(2,5),  BIC_PN_LNorm, c='b', label='LogNormal' )
plt.plot(np.arange(2,11), BIC_PN_Logis, c='g', label='Logis'     )

plt.xlim(1, 11)

plt.xlabel('States', fontsize=14)
plt.ylabel('BIC', fontsize=14)

plt.xticks(size=13)
plt.yticks(size=13)

plt.title("BIC - PN's HMM", fontsize=14)
plt.grid('True')
plt.legend(loc='right center')

plt.savefig('/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/Validation/PN_BIC.png', bbox_inches='tight', dpi=300)
plt.close('all')



