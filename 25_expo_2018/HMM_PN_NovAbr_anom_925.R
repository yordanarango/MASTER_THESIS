library(HiddenMarkov)
library(fitdistrplus)


###############################   Nov_Abr   ################################

data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_PN_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd = c(data$V1)

"Necesitamos que esté entre [0, 1) porque se va a utilizar la distribución beta"
spd = spd/max(spd) * 0.9999999 # Se multiplica por 0.9999999 porque la serie no puede tomar el valor de 1, porque luego no es posible calcular la bondad de ajuste

fl <- fitdist(spd, "beta", method="mge")  # beta: los valores deben ser dados entre 0 y 1

"2 Estados"
Pi    <- rbind(c(0.7, 0.3), c(0.4, 0.6))
delta <- c(1,0)
pm    <- list(shape1=c(fl$estimate[1], fl$estimate[1]), 
              shape2=c(fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "beta", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "beta", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE
states2 <- Viterbi(x)

"3 Estados"
Pi    <- rbind(c(0.7, 0.2, 0.1), c(0.3, 0.6, 0.1), c(0.2, 0.2, 0.6))
delta <- c(1,0,0)
pm    <- list(shape1=c(fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              shape2=c(fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "beta", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "beta", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE
states3 <- Viterbi(x)

"4 Estados"
Pi    <- rbind(c(0.8, 0.1, 0.05, 0.05), c(0.1, 0.6, 0.25, 0.05), c(0.1, 0.15, 0.5, 0.25), c(0.2, 0.05, 0.2, 0.55))
delta <- c(1,0,0,0)
pm    <- list(shape1=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              shape2=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "beta", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "beta", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE
states4 <- Viterbi(x)



"5 Estados"
Pi    <- rbind(c(0.7, 0.1, 0.05, 0.05, 0.1), c(0.05, 0.6, 0.25, 0.05, 0.05), c(0.1, 0.1, 0.5, 0.25, 0.05), c(0.2, 0.05, 0.15, 0.55, 0.05), c(0.05, 0.3, 0.1, 0.05, 0.5))
delta <- c(1,0,0,0,0)
pm    <- list(shape1=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              shape2=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "beta", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "beta", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE
states5 <- Viterbi(x)



"6 Estados"
Pi    <- rbind(c(0.65, 0.1, 0.05, 0.05, 0.07, 0.08), c(0.02, 0.6, 0.2, 0.08, 0.05, 0.05), c(0.1, 0.15, 0.5, 0.15, 0.05, 0.05), c(0.07, 0.13, 0.05, 0.15, 0.55, 0.05), c(0.05, 0.23, 0.1, 0.05, 0.5, 0.07), c(0.15, 0.1, 0.03, 0.07, 0.15, 0.5))
delta <- c(1,0,0,0,0,0)
pm    <- list(shape1=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              shape2=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "beta", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "beta", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- TRUE
states6 <- Viterbi(x)


"Datframe"

DF <- data.frame("states2" = states2, "states3" = states3, "states4" = states4, "states5" = states5, "states6" = states6)
write.csv(DF, 
          file = "/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/States_PN_NovAbr_anom_925.csv")

