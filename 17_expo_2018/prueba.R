library(msm)

data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/17_expo_2018/datos_PN_EneDic.csv"
                 , header=FALSE, sep=",")

spd     = c(data$V1)
time    = c(data$V2)
ptnum   = rep(1, length(spd))
DF      = data.frame(spd, time, ptnum)



HMM2_PN = msm(spd~time, subject=ptnum, data=DF) 












