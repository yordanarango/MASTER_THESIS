library(HiddenMarkov)
library(fitdistrplus)


###############################   Nov_Abr   ################################

data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/datos_TT_NovAbr_anom_925_6h.csv"
                 , header=FALSE, sep=",")

spd = c(data$V1)

fl <- fitdist(spd, "gamma")

"2 Estados"
Pi    <- rbind(c(0.7, 0.3), 
               c(0.4, 0.6))
delta <- c(0,1)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "gamma", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states2 <- Viterbi(x)


"3 Estados"
Pi    <- rbind(c(0.7, 0.2, 0.1), 
               c(0.3, 0.6, 0.1), 
               c(0.2, 0.2, 0.6))
delta <- c(0,0,1)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "gamma", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states3 <- Viterbi(x)


"4 Estados"
Pi    <- rbind(c(0.62, 0.14, 0.13, 0.11), 
               c(0.1,  0.55, 0.2,  0.15), 
               c(0.1,  0.15, 0.5,  0.25), 
               c(0.2,  0.1,  0.15,  0.55))
delta <- c(0,0,0,1)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate =c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "gamma", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states4 <- Viterbi(x)



"5 Estados"
Pi    <- rbind(c(0.55, 0.1,  0.12, 0.13, 0.1), 
               c(0.11, 0.51, 0.19, 0.09, 0.1), 
               c(0.1,  0.1,  0.5,  0.19, 0.11), 
               c(0.1,  0.12, 0.12, 0.48, 0.18), 
               c(0.12, 0.13, 0.1,  0.15, 0.5))

delta <- c(0,0,0,0,1)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "gamma", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states5 <- Viterbi(x)



"6 Estados"
Pi    <- rbind(c(0.4,  0.14, 0.12,  0.1,  0.1,   0.14), 
               c(0.11, 0.38, 0.1,   0.13, 0.12,  0.16), 
               c(0.1,  0.15, 0.41,  0.15, 0.09,  0.1), 
               c(0.12, 0.13, 0.11,  0.4,  0.14,  0.1), 
               c(0.1,  0.18, 0.1,   0.12, 0.39,  0.11), 
               c(0.15, 0.1,  0.1,   0.12, 0.15,  0.38))

delta <- c(0,0,0,0,0,1)
pm    <- list(shape=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              rate=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "gamma", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states6 <- Viterbi(x)


"Datframe"

DF <- data.frame("states2" = states2, "states3" = states3, "states4" = states4, "states5" = states5, "states6" = states6)
write.csv(DF, 
          file = "/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/27_expo_2018/States_TT_NovAbr_anom_925.csv")

