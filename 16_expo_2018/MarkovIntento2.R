#MODELO DE MARKOV OCULTO APLICADO A CAUDALES MENSUALES Y DIARIOS--------------------

#Analizar estados
#-----Ajustar modelo
#-----Max Log(Maximaverosimilitud)
#Simular Caudales

#Profesor Oscar Jóse Mesa

#Juan Daniel González Caldera

#Fractales y caos [Sistemas Dinámicos Aleatorios]

#Comentarios
# Beta ("beta"), Binomial ("binom"), Exponential ("exp"), GammaDist ("gamma"), 
#Lognormal ("lnorm"), Logistic ("logis"), Normal ("norm"), and Poisson ("pois").

#Librarías-----------------------------------------------------------------------------

library(mise)
mise()
library(readr)
library(HiddenMarkov)
library(raster)
library(plot3D)
library(fitdistrplus)

#Input---------------------------------------------------------------------------------
Caudales_diarios <- read_delim("~/Fractales y caos/Caudales diarios.txt", 
                               "\t", escape_double = FALSE, trim_ws = TRUE)
Caudales <- as.data.frame(Caudales_diarios)
DataNaN <- is.na(Caudales[,1])
Caudales <- Caudales[DataNaN == FALSE,]
#Organizar los meses-------------------------------------------------------------------
#Resultados Dataframe QDiaMes, tiene los siguientes meses
MesesAnalisis <- seq(9,11,1) #Agosto, Septiembre y Octubre

cont1 <- 0

for (i in 1:nrow(Caudales)) {
  if (Caudales[i,2] >= MesesAnalisis[1] & Caudales[i,2] <= MesesAnalisis
      [length(MesesAnalisis)]) {
    cont1 <- cont1 + 1
  }
  if (Caudales[i,2] > MesesAnalisis[length(MesesAnalisis)]) {
    break
  }
}

NumDays <- cont1
NumYear <- -(Caudales[1,1] - Caudales[nrow(Caudales),1]) + 1

QDiaMes <- matrix(NaN,NumYear*NumDays,4)
QDiaMes <- as.data.frame(QDiaMes)

cont2 <- 0
for (i in 1:nrow(Caudales)) {
  if (Caudales[i,2] >= MesesAnalisis[1] & Caudales[i,2] <= 
      MesesAnalisis[length(MesesAnalisis)]) {
    cont2 <- cont2 + 1
    QDiaMes[cont2,c(1,2,3,4)] <- Caudales[i,c(1,2,3,4)]
  }
}

DataMesNaN <- is.na(QDiaMes[,1])
QDiaMes <- QDiaMes[DataMesNaN == FALSE,]
#Caudales Mensuales--------------------------------------------------------------------
#Llevar diarios a mensuales
NumMeses <- NumYear*12
CaudalesMensuales <- matrix(NaN,NumMeses,3)
CaudalesMensuales <- as.data.frame(CaudalesMensuales)
colnames(CaudalesMensuales) <- c("Year","Meses","Caudal")

Cont3 <- 1
Cont4 <- 0
Cont5 <- 0
Qmensual <- Caudales[1,4]
for (i in 1:(nrow(Caudales) - 1)) {
  
  if (Caudales[i,2] == Caudales[i + 1,2]) {
    if (is.na(Caudales[i + 1,4])) { 
    } else {
      Cont3 <- Cont3 + 1
      Qmensual <- Qmensual + Caudales[i + 1,4] 
    }
  } else {
    Cont4 <- Cont4 + 1
    
    if (Cont3 > 20) { 
      CaudalesMensuales[Cont4,3] <- Qmensual / Cont3
      CaudalesMensuales[Cont4,2] <- Caudales[i,2]
      CaudalesMensuales[Cont4,1] <- Caudales[i,1]
      
      if (is.na(Caudales[i,4])) {
        Cont3 <- 0
        Qmensual <- 0
      } else {
        Cont3 <- 1
        Qmensual <- Caudales[i + 1,4]
      }
    } else {
      CaudalesMensuales[Cont4,3] <- NaN
      CaudalesMensuales[Cont4,2] <- Caudales[i,2]
      CaudalesMensuales[Cont4,1] <- Caudales[i,1]
      
      
      if (is.na(Caudales[i,4])) {
        Cont3 <- 0
        Qmensual <- 0
      } else {
        Cont3 <- 1
        Qmensual <- Caudales[i + 1,4]
      }
    }

  }
}

Cont4 <- Cont4 + 1
if (Cont3 > 20) { 
  CaudalesMensuales[Cont4,3] <- Qmensual / Cont3
  CaudalesMensuales[Cont4,2] <- Caudales[i,2]
  CaudalesMensuales[Cont4,1] <- Caudales[i,1]
} else {
  CaudalesMensuales[Cont4,3] <- NaN
  CaudalesMensuales[Cont4,2] <- Caudales[i,2]
  CaudalesMensuales[Cont4,1] <- Caudales[i,1]
}

plot(Caudales[,1] + Caudales[,2]/12 + Caudales[,3]/(31*12),Caudales[,4],type =
       "l", col = "steelblue2", xlab = "",ylab="Caudal [m3/s]",
     main = "Río Cauca -  Estación Cañafisto", ylim = c(00,3500))
lines(CaudalesMensuales[,1] + CaudalesMensuales[,2]/12,CaudalesMensuales[,3],
      type = "l",col = "red")
legend("topleft",c("Caudales diarios","Caudales mensuales"),lwd = 1
       , col = c("steelblue2","red"), box.col = "white", inset = 0.00625)
#Ciclo Anual--------------------------------------------------------------------
#Para comparar con el modelo mensual
MatrixMensual <- matrix(NaN,NumYear,13)
Cont11 <- 0
MatrixMensual[,1] <- seq(CaudalesMensuales[1,1],CaudalesMensuales[1,1] + NumYear
                         -1)
for (i in 1:NumYear) {
  for (j in 1:12) {
    Cont11 <- Cont11 + 1
    MatrixMensual[i,j+1] <- CaudalesMensuales[Cont11,3]
  }
}

MatrixGraf <- MatrixMensual[,2:13]
MatrixGraf <- t(MatrixGraf)
image2D(log(MatrixGraf),border = "black")

ciclo <- matrix(NaN,5,13)

ciclo[2,] <- MatrixMensual[MatrixMensual[,1] == 1982,]
ciclo[3,] <- MatrixMensual[MatrixMensual[,1] == 1992,]
ciclo[4,] <- MatrixMensual[MatrixMensual[,1] == 2000,]
ciclo[5,] <- MatrixMensual[MatrixMensual[,1] == 2008,]

for (i in 1:12) {
  ciclo[1,i+1] <- mean(MatrixMensual[,i+1],na.rm = TRUE)
}

plot(seq(1,12),ciclo[1,2:13],type = "l",main = "Ciclo Anual",ylab =
       "Caudal [m3/s]"
     ,xlab = "",col = "blue")

#Datos sin vacios---------------------------------------------------------------
#Quitar los no datos de las series
NaNQD <- is.na(QDiaMes[,4])
DatoDias <- QDiaMes[NaNQD == FALSE,]
NaNMes <- is.na(CaudalesMensuales[,3])
DatosMensuales <- CaudalesMensuales[NaNMes == FALSE,]

#Distribuciones-----------------------------------------------------------------
#Distribución de emisión a escoger fit
try(fg <- fitdist(DatosMensuales[,3], "gamma"), fg <- 0)     #Gamma
try(fln <- fitdist(DatosMensuales[,3], "lnorm"), fln <- 0)   #Lognormal
try(flog <- fitdist(DatosMensuales[,3], "logis"), flog <- 0) #logística
par(mfrow = c(1, 1))

#Gráficas de los ajustes
listdis <- list(fg,fln,flog)
plot.legend <- c("Gamma","Lognormal","Logística")
denscomp(listdis, legendtext = plot.legend, xlab = "Caudal [m3/s]", main = 
           "Histograma de caudales mensuales", ylab = "Densidad de probabilidad"
         ,cex.main = 1.6, cex.lab = 1.5,cex.axis = 1.4)
legend("topright",c("Logística","Gamma","Lognormal"),lwd = c(1,1,1)
       ,lty = c(3,1,2), col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

qqcomp(listdis, legendtext = plot.legend,xlab = "teóricos",ylab ="Empíricos",
       main = "Cuantil - Cuantil",cex.main = 1.6,cex.lab = 1.5,cex.axis = 1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),pch = c(1,1,1)
       , col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

cdfcomp(listdis, legendtext = plot.legend, main = "Distribución acumulada",
        xlab = "Caudal [m3/s]",cex.main =1.6,cex.lab = 1.5,cex.axis =1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),lwd = c(1,1,1)
       ,lty = c(3,1,2), col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

ppcomp(listdis, legendtext = plot.legend, main = "P - P", ylab = "Empíricas",
       xlab = "Teóricas",cex.main = 1.6,cex.lab=1.5,cex.axis = 1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),pch = c(1,1,1)
       , col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

kstestM <- gofstat(list(fg, fln, flog))

try(fgd <- fitdist(DatoDias[,4], "gamma"), fgd <- 0)     #Gamma
try(flnd <- fitdist(DatoDias[,4], "lnorm"), flnd <- 0)   #Lognormal
try(flogd <- fitdist(DatoDias[,4], "logis"), flogd <- 0) #logística
par(mfrow = c(1, 1))

listdis <- list(fgd,flnd,flogd)
plot.legend <- c("Gamma","Lognormal","Logística")
denscomp(listdis, legendtext = plot.legend, xlab = "Caudales [m3/s]", main = 
           "Histograma de caudales diarios", ylab = "Densidad de probabilidad"
         ,cex.main = 1.6,
         cex.lab = 1.5, cex.axis = 1.4)
legend("topright",c("Logística","Gamma","Lognormal"),lwd = c(1,1,1)
       ,lty = c(3,1,2), col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

qqcomp(listdis, legendtext = plot.legend,xlab = "teóricos",ylab ="Empíricos",
       main = "Cuantil - Cuantil",cex.main = 1.6,cex.lab=1.5,cex.axis=1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),pch = c(1,1,1)
       , col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

cdfcomp(listdis, legendtext = plot.legend, main = "Distribución acumulada",
        xlab = "Caudales [m3/s]",cex.main=1.6,cex.lab=1.5,cex.axis=1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),lwd = c(1,1,1)
       ,lty = c(3,1,2), col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

ppcomp(listdis, legendtext = plot.legend, main = "P - P", ylab = "Empíricas",
       xlab = "Teóricas",cex.main=1.6,cex.lab=1.5,cex.axis=1.4)
legend("bottomright",c("Logística","Gamma","Lognormal"),pch = c(1,1,1)
       , col = c("blue","red","green"), box.col = "white",
       inset = 0.00625,cex = 1.3)

kstestD <- gofstat(list(fgd, flnd, flogd))
#Modelo de Markov Diario--------------------------------------------------------
DiasOrden <- sort(DatoDias[,4])
ndias <- nrow(DatoDias)
NegLog <- matrix(NaN,9-3+1,2)
NegLog <- as.data.frame(NegLog)
colnames(NegLog) <- c("Estados","Log")
Cont9 <- 0
for (i in 3:9) {
  Cont9 <- Cont9 + 1
  Pi <- matrix(0,i,i)
  diagonal <- cbind(seq(1,i),seq(1,i))
  diagonalsuperior <- cbind(seq(1,i-1),seq(2,i))
  diagonalinferior <- cbind(seq(2,i),seq(1,i-1))
  Pi[diagonal] <- 0.6
  Pi[diagonalinferior] <- 0.2
  Pi[diagonalsuperior] <- 0.2
  Pi[1,2] <- 0.4
  Pi[i,i - 1] <- 0.4
  
  delta <- rep(0,i)
  delta[1] <- 1
  
  PartePara <- seq(1,floor(ndias/i)*i,floor(ndias/i)-1)
  Media <- matrix(NaN,i,1)
  Desv <- matrix(NaN,i,1)
  for (j in 1:i) {
    distEstado <- fitdist(DiasOrden[seq(PartePara[j],PartePara[j+1])],"lnorm")
    Media[j,1] <- distEstado$estimate[1]
    Desv[j,1] <- distEstado$estimate[2]
  }
  
  pm <- list(Media,Desv)
  x <- dthmm(NULL, Pi, delta, "lnorm", pm , discrete = FALSE)
  x$x <- DatoDias[,4]
  x$nonstat <- FALSE
  states <- Viterbi(x)
  x$y <- states
  
  options(warn = -1)
  y <- BaumWelch(x)
  NegLog[Cont9,2] <- logLik(y)
  NegLog[Cont9,1] <- i
}

EstadosQdias <- matrix(NaN,nrow(QDiaMes),1)
EstadosQdias <- as.data.frame(EstadosQdias)

EstadosQdias[NaNQD == FALSE,] <- y$y

MatrizEstados <- matrix(NaN,NumYear,NumDays)
MatrizEstados <- as.data.frame(MatrizEstados)

Cont6 <- 0
for (i in 1:NumYear) {
  for (j in 1:NumDays) {
    Cont6 <- Cont6 + 1
    MatrizEstados[i,j] <- EstadosQdias[Cont6,1]
  }
}

MatrizEstados <- t(MatrizEstados)


date <- paste(QDiaMes[,1],"/",QDiaMes[,2],"/",QDiaMes[,3],sep = "")
YYear <- seq(QDiaMes[1,1],QDiaMes[nrow(QDiaMes),1],1)
dategraf <- date[1:(NumDays + 1)]
dategraf[92] <- "1979/12/01"
xgraf <- as.Date(dategraf)
par(mfrow = c(1, 1))

plot.new()

image2D(MatrizEstados,y = YYear,border = "black",x = seq(1,92),main 
     ="Matriz de estados",xlab = "Septiembre - Octubre - Noviembre", ylab = "")
# axis.Date(1,at = 1:92,x = xgraf, format = "%m-%d")
#Modelo de Markov Mensuales-----------------------------------------------------
MesOrden <- sort(DatosMensuales[,3])
nMes <- nrow(DatosMensuales)
NegLog <- matrix(NaN,5-5+1,2)
NegLog <- as.data.frame(NegLog)
colnames(NegLog) <- c("Estados","Log")
Cont9 <- 0
for (i in 5:5) {
  Cont9 <- Cont9 + 1
  Pi <- matrix(0,i,i)
  diagonal <- cbind(seq(1,i),seq(1,i))
  diagonalsuperior <- cbind(seq(1,i-1),seq(2,i))
  diagonalinferior <- cbind(seq(2,i),seq(1,i-1))
  Pi[diagonal] <- 0.6
  Pi[diagonalinferior] <- 0.2
  Pi[diagonalsuperior] <- 0.2
  Pi[1,2] <- 0.4
  Pi[i,i - 1] <- 0.4
  
  delta <- rep(0,i)
  delta[1] <- 1
  
  PartePara <- seq(1,floor(nMes/i)*i,floor(nMes/i)-1)
  Media <- matrix(NaN,i,1)
  Desv <- matrix(NaN,i,1)
  for (j in 1:i) {
    distEstado <- fitdist(MesOrden[seq(PartePara[j],PartePara[j+1])],"lnorm")
    Media[j,1] <- distEstado$estimate[1]
    Desv[j,1] <- distEstado$estimate[2]
  }
  
  pm <- list(Media,Desv)
  x <- dthmm(NULL, Pi, delta, "lnorm", pm , discrete = FALSE)
  x$x <- DatosMensuales[,3]
  x$nonstat <- FALSE
  states <- Viterbi(x)
  x$y <- states
  
  options(warn = -1)
  y <- BaumWelch(x)
  NegLog[Cont9,2] <- logLik(y)
  NegLog[Cont9,1] <- i
}

options(warn = 3)

EstadosQmensual <- matrix(NaN,nrow(CaudalesMensuales),1)
EstadosQmensual <- as.data.frame(EstadosQmensual)

EstadosQmensual[NaNMes == FALSE,] <- y$y

MatrizEstados <- matrix(NaN,NumYear,12)
MatrizEstados <- as.data.frame(MatrizEstados)

Cont10 <- 0
for (i in 1:NumYear) {
  for (j in 1:12) {
    Cont10 <- Cont10 + 1
    MatrizEstados[i,j] <- EstadosQmensual[Cont10,1]
  }
}

MatrizEstados <- t(MatrizEstados)

image2D(MatrizEstados, y = YYear,x = seq(1,12),border = "black",
        main = "Matriz de estado mensual",ylab="",xlab="")
#Simulación de caudales mensuales-----------------------------------------------

nsimul <- 240
iter <- 1500
Simulall <- matrix(NaN,nsimul,iter)
error <- matrix(NaN,1,iter)
for (j in 1:iter) {
yeari <- 2005
yearf <- yeari + nsimul/12 - 1
Simulada <- matrix(NaN,(yearf - yeari + 1)*12,3)
Simulada[,2] <- rep(seq(1,12),(yearf - yeari + 1))
yearC <- yeari - 1
for (i in 1:nrow(Simulada)) {
  if (Simulada[i,2] == 1){
    yearC <- yearC + 1
  }
  Simulada[i,1] <- yearC
}

simulMen <- simulate(y,nsim = nsimul)
Simulada [,3] <- simulMen$x
Simulall[,j] <- Simulada[,3]

error[1,j] <- sum((Simulada[1:72,3] - CaudalesMensuales[(nrow(CaudalesMensuales)
                                                         - 71):
                   nrow(CaudalesMensuales),3]) ^ 2,na.rm = TRUE)
}
Simulada[,3] <- Simulall[,error == min(error)]
plot(CaudalesMensuales[,1]+CaudalesMensuales[,2]/12, CaudalesMensuales[,3],
     type = "l", col = "blue", xlim = c(CaudalesMensuales[1,1],yearf), main = 
       "Simulación de caudales mensuales", xlab = "",ylab = "Caudal [m3/s]")
lines(Simulada[,1] + Simulada[,2]/12,Simulada[,3],type ="l", col = "red")
#ONI (Oceanic Niño Index)-------------------------------------------------------
#Se calcula como la media móvil de tres meses de las anomalías de la 
#temperatura superficial del mar (SST) en la región Niño 3.4 (5N-5S, 120-70W), 
#la identificación de meses cálidos (anomalías positivas, El Niño) o fríos 
#(anomalías negativas, La Niña), se da cuando el valor del ONI supera el umbral
#de +0.5°C para El Niño o es inferior a -0.5°C para La Niña. Para efectos 
#históricos se dice que el episodio fue cálido o frio cuando lleva por lo menos
#5 meses consecutivos en dichos umbrales. 

#Se calcula para el análisis de resultado
DatosONI <- read.csv("~/Fractales y caos/ONI.txt", sep="")

MesONI <- matrix(NaN,nrow(DatosONI),1)

#Mes Inicial el valor del Cont8
Cont8 <- 0
for (i in 1:nrow(DatosONI)) {
  Cont8 <- Cont8 + 1
  MesONI[i,1] <- Cont8
  if (Cont8 == 12) {
    Cont8 <- 0
  }
}

ONIv <- cbind(DatosONI[,2],MesONI[,1],DatosONI[,4])
ONIvector <- matrix(NaN,nrow(ONIv),5)
ONIvector[,c(1,2,3)] <- ONIv
  
ValorPositivo <- ONIvector[ONIvector[,3] > 0,3]
ONIvector[ONIvector[,3] > 0 | is.na(ONIvector[,3]),4] <- ValorPositivo
ValorNegativo <- ONIvector[ONIvector[,3] < 0,3]
ONIvector[ONIvector[,3] < 0 | is.na(ONIvector[,3]),5] <- ValorNegativo 

plot(ONIvector[,1] + ONIvector[,2]/12,ONIvector[,4],type="h",col ="red",
     ylim = c(min(ONIvector[,3],na.rm = TRUE),max(ONIvector[,3],na.rm = TRUE)),
     xlim = c(min(ONIvector[,1],na.rm = TRUE),2018), xlab = "",
     main = "Oceanic Niño Index", ylab = "ONI Index °C")
lines(ONIvector[,1] + ONIvector[,2]/12,ONIvector[,5],type="h",col ="dodgerblue")
lines(c(1940,2050),c(0.5,0.5),type="l",col = "black")
lines(c(1940,2050),c(-0.5,-0.5),type="l",col = "black")

CorrONI <- ONIvector[min(CaudalesMensuales[,1]) <= ONIvector[,1] & 
                       max(CaudalesMensuales[,1]) >= ONIvector[,1],3]

VCorrONI <- cbind(CaudalesMensuales,CorrONI)

CorrONI <- acf(VCorrONI[,c(3,4)],"correlation",na.action = na.pass,lag.max = 6)
#SOI (Southern Oscillation Index)-----------------------------------------------
#Los cambios en la circulación de los vientos Alisios debidos a perturbaciones 
#de la dinámica atmosférica relacionados con cambios en los sistemas de presión 
#a nivel del mar se conoce como Oscilación Sur. El índice se mide como la 
#anomalía de la diferencia de la presión atmosférica al nivel del mar media 
#mensual entre Tahití (Polinesia Francesa) y Darwin (Norte de Australia), 
#valores negativos del SOI indican que la presión en el este disminuyo y está 
#asociado al El Niño debido al calentamiento del este del mar pacífico que 
#provoca que la atmósfera se expanda y viceversa para La Niña, en condiciones 
#normales el SOI muestra valores positivos debido a que la piscina de agua 
#caliente se encuentra en Australia, pero cuando ocurre La Niña el gradiente de 
#presión se intensifica. 

#Se calcula para el análisis de resultados
DatosSOI <- read.csv("~/Fractales y caos/SOI.txt", sep="")

SOI <- matrix(NaN,nrow(DatosSOI),ncol(DatosSOI))

NaNSOI <- DatosSOI == -999.9
SOI[NaNSOI == FALSE] <- DatosSOI[NaNSOI == FALSE]

nyearSOI <- nrow(SOI)

SOIvector <- matrix(NaN,nyearSOI*12,5)

Cont7 <- 0
for (i in 1:nyearSOI) { 
  for (j in 1:12) {
    Cont7 <- Cont7 + 1
    SOIvector[Cont7,3] <- SOI[i,j + 1]
    SOIvector[Cont7,2] <- j
    SOIvector[Cont7,1] <- SOI[i,1]
  }
}

ValorPositivo <- SOIvector[SOIvector[,3] > 0,3]
SOIvector[SOIvector[,3] > 0 | is.na(SOIvector[,3]),4] <- ValorPositivo
ValorNegativo <- SOIvector[SOIvector[,3] < 0,3]
SOIvector[SOIvector[,3] < 0 | is.na(SOIvector[,3]),5] <- ValorNegativo

plot(SOIvector[,1] + SOIvector[,2]/12,SOIvector[,4],type="h",col ="dodgerblue",
     ylim = c(min(SOIvector[,3],na.rm = TRUE),max(SOIvector[,3],na.rm = TRUE)),
     xlim = c(min(SOIvector[,1],na.rm = TRUE),2018), xlab = "",
     main = "Southern Oscillation Index", ylab = "SOI")
lines(SOIvector[,1] + SOIvector[,2]/12,SOIvector[,5],type="h",col ="red")

CorrSOI <- SOIvector[min(CaudalesMensuales[,1]) <= SOIvector[,1] & 
                       max(CaudalesMensuales[,1]) >= SOIvector[,1],3]

VCorrSOI <- cbind(CaudalesMensuales,CorrSOI)

CorrSOI <- acf(VCorrSOI[,c(3,4)],"correlation",na.action = na.pass,lag.max = 6)

