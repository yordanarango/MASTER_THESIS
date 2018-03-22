library(HiddenMarkov)
library(fitdistrplus)
library(ncdf4)

############################# TEHUANTEPEC ###############################################
archivo = nc_open('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
time    = ncvar_get(archivo,"time")

U       = ncvar_get(archivo,"u10")[33, 9, seq(4,length(time),by=4)]
V       = ncvar_get(archivo,"v10")[33, 9, seq(4,length(time),by=4)]
spd     = sqrt(U*U+V*V)

"Ajustes"
dgm <- fitdist(spd, "gamma")         # gamma
dln <- fitdist(spd, "lnorm")         # lognormal
dlg <- fitdist(spd, "logis")         # logistica
dwb <- fitdist(spd, "weibull")       # weibull

"Graficas todos"
listdis     <- list(dgm, dln, dlg, dwb)
plot.legend <- c("Gamma", "Lognormal", "Logistic", "Weibull")
denscomp(listdis, legendtext = plot.legend, xlab = "Speed [m/s]", main = 
           "TT - Speed PDF", ylab = "PDF", 
         fitcol=c("blue","red","goldenrod", "dimgray"), 
         fitlty=c(1,1,1,1), ylim=c(0,0.5))

legend("topright",c("Gamma", "Lognormal", "Logistic", "Weibull"), 
       lwd = c(1,1,1),
       col = c("blue", "red", "goldenrod", "dimgray"), box.col = "white",
       inset = 0.00625, cex = 1.3, lty=c(1,1,1,1))


listdis     <- list(dwb)
plot.legend <- c("Weibull")
denscomp(listdis, legendtext = plot.legend, xlab = "Speed [m/s]", main = 
           "TT - Speed PDF", ylab = "PDF", 
         fitcol=c("dimgray"), fitlty=c(1), ylim=c(0,0.5))

legend("topright",c("Weibull"), lwd = c(1),
       box.col = "white",
       inset = 0.00625, cex = 1.3, 
       col = c("dimgray"), lty=c(1))

#############################  PANAMA  #############################################
data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/16_expo_2018/datos_PN_distribuciones.csv"
                 , header=FALSE, sep=",")

spd = c(data$V1)

"Ajustes"
dgm <- fitdist(spd, "gamma")         # gamma
dln <- fitdist(spd, "lnorm")         # lognormal
dlg <- fitdist(spd, "logis")         # logistica
dwb <- fitdist(spd, "weibull")       # weibull

"Graficas todos"
listdis     <- list(dgm, dln, dlg)
plot.legend <- c("Gamma", "Lognormal", "Logistic")
denscomp(listdis, legendtext = plot.legend, xlab = "Speed [m/s]", main = 
           "PN - Speed PDF", ylab = "PDF", 
         fitcol=c("blue","red","green"), 
         fitlty=c(1,2,3))

legend("topright",c("Gamma", "Lognormal", "Logistic"), 
       lwd = c(1,1,1),
       col = c("blue","red","green"), box.col = "white",
       inset = 0.00625, cex = 1.3, lty=c(1,2,3))

"Graficas individual"
listdis     <- list(dwb)
plot.legend <- c("Weibull")
denscomp(listdis, legendtext = plot.legend, xlab = "Speed [m/s]", main = 
           "PN - Speed PDF", ylab = "PDF", 
         fitcol=c("darkgreen"), fitlty=c(4))

legend("topright",c("Weibull"), lwd = c(1),
       box.col = "white",
       inset = 0.00625, cex = 1.3, 
       col = c("darkgreen"), lty=c(4))
