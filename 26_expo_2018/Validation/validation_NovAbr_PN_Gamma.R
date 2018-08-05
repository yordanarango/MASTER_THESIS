library(HiddenMarkov)
library(fitdistrplus)
library(beepr)

"Lectura de datos"
data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_PN_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd     = c(data$V1)
spd     = spd/max(spd) * 0.9999999 # Se multiplica por 0.9999999 porque la serie no puede tomar el valor de 1, porque luego no es posible calcular la bondad de ajuste
#spd_cal = spd[1:5075]        #Datos de calibración. Hasta 2007-04-30
#spd_val = tail(spd, n=-5075) #Datos de validación. A partir de 2007-11-01

"Gamma"
fl <- fitdist(spd, "gamma")

"2 Estados"
Pi    <- rbind(c(0.7, 0.3), 
               c(0.4, 0.6))

delta <- c(1,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2]))

x         <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x       <- spd
x$nonstat <- FALSE

y      <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
beep()
print(y$LL)



"3 Estados"
Pi    <- rbind(c(0.7, 0.2, 0.1), 
               c(0.3, 0.6, 0.1), 
               c(0.2, 0.2, 0.6))

delta <- c(1,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
beep()
print(y$LL)



"4 Estados"
Pi    <- rbind(c(0.8, 0.1,  0.05, 0.05), 
               c(0.1, 0.6,  0.25, 0.05), 
               c(0.1, 0.15, 0.5,  0.25), 
               c(0.2, 0.05, 0.2,  0.55))

delta <- c(1,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate =c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y     <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"5 Estados"
Pi    <- rbind(c(0.7,  0.1,  0.05, 0.05, 0.1), 
               c(0.05, 0.6,  0.25, 0.05, 0.05), 
               c(0.1,  0.1,  0.5,  0.25, 0.05), 
               c(0.2,  0.05, 0.15, 0.55, 0.05), 
               c(0.05, 0.3,  0.1,  0.05, 0.5))

delta <- c(1,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y       <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"6 Estados"
Pi    <- rbind(c(0.65, 0.1,  0.05, 0.05, 0.07, 0.08), 
               c(0.02, 0.6,  0.2,  0.08, 0.05, 0.05), 
               c(0.1,  0.15, 0.5,  0.15, 0.05, 0.05), 
               c(0.07, 0.13, 0.05, 0.55, 0.15, 0.05), 
               c(0.05, 0.23, 0.1,  0.05, 0.5,  0.07), 
               c(0.15, 0.1,  0.03, 0.07, 0.15, 0.5))

delta <- c(1,0,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"7 estados"
Pi    <- rbind(c(0.65, 0.1,  0.05, 0.02, 0.05, 0.07, 0.06), 
               c(0.02, 0.4,  0.2,  0.08, 0.05, 0.2,  0.05), 
               c(0.1,  0.1,  0.5,  0.15, 0.05, 0.05, 0.05), 
               c(0.03, 0.07, 0.13, 0.52, 0.05, 0.15, 0.05), 
               c(0.05, 0.2,  0.1,  0.03, 0.5,  0.05, 0.07), 
               c(0.15, 0.1,  0.1,  0.03, 0.07, 0.4,  0.15),
               c(0.17, 0.2,  0.05, 0.03, 0.07, 0.15, 0.33))

delta <- c(1,0,0,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"8 estados"
Pi    <- rbind(c(0.45, 0.1,  0.07,  0.12, 0.03, 0.07, 0.07, 0.09), 
               c(0.02, 0.4,  0.15,  0.05, 0.08, 0.05, 0.2,  0.05), 
               c(0.1,  0.1,  0.5,   0.07, 0.08, 0.05, 0.05, 0.05), 
               c(0.02, 0.03, 0.07,  0.5,  0.13, 0.05, 0.15, 0.05), 
               c(0.05, 0.2,  0.1,   0.03, 0.3,  0.05, 0.07, 0.2), 
               c(0.13, 0.1,  0.05,  0.1,  0.03, 0.4,  0.07, 0.12),
               c(0.1,  0.1,  0.1,   0.03, 0.07, 0.15, 0.33, 0.12),
               c(0.13, 0.07, 0.1,   0.02, 0.03, 0.15, 0.08, 0.42))

delta <- c(1,0,0,0,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"9 estados"
Pi    <- rbind(c(0.36, 0.09,  0.075, 0.13, 0.04, 0.08, 0.07, 0.09, 0.065), 
               c(0.02, 0.4,   0.13,  0.05, 0.02, 0.08, 0.05, 0.2,  0.05), 
               c(0.07, 0.07,  0.5,   0.07, 0.06, 0.08, 0.05, 0.05, 0.05), 
               c(0.02, 0.03,  0.07,  0.35, 0.13, 0.05, 0.1,  0.05, 0.2), 
               c(0.05, 0.25,  0.05,  0.03, 0.3,  0.05, 0.07, 0.18, 0.02), 
               c(0.13, 0.09,  0.05,  0.1,  0.03, 0.4,  0.07, 0.1,  0.03),
               c(0.1,  0.05,  0.12,  0.03, 0.07, 0.15, 0.3,  0.07, 0.11),
               c(0.13, 0.07,  0.1,   0.08, 0.02, 0.03, 0.12, 0.37, 0.08),
               c(0.11, 0.07,  0.09,  0.08, 0.02, 0.03, 0.11, 0.06, 0.43))

delta <- c(1,0,0,0,0,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)



"10 estados"
Pi    <- rbind(c(0.3, 0.06,  0.07,  0.14, 0.09, 0.04,  0.08,  0.07, 0.09, 0.06), 
               c(0.07, 0.35,  0.02,  0.12, 0.04, 0.02,  0.08,  0.05, 0.2,  0.05), 
               c(0.07, 0.05,  0.48,  0.07, 0.06, 0.085, 0.065, 0.02, 0.05, 0.05), 
               c(0.02, 0.03,  0.07,  0.34, 0.13, 0.05,  0.1,   0.05, 0.2,  0.01), 
               c(0.05, 0.23,  0.05,  0.03, 0.3,  0.05,  0.07,  0.15, 0.05, 0.02), 
               c(0.11, 0.09,  0.05,  0.1,  0.03, 0.37,  0.07,  0.1,  0.03, 0.05),
               c(0.1,  0.05,  0.12,  0.03, 0.07, 0.14,  0.29,  0.07, 0.11, 0.02),
               c(0.1,  0.09,  0.07,  0.1,  0.07, 0.02,  0.03,  0.33, 0.11, 0.08),
               c(0.11, 0.07,  0.1,   0.09, 0.08, 0.02,  0.03,  0.11, 0.33, 0.06),
               c(0.09, 0.17,  0.05,  0.11, 0.04, 0.01,  0.03,  0.09, 0.05, 0.36))

delta <- c(1,0,0,0,0,0,0,0,0,0)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
beep()
print(y$LL)


