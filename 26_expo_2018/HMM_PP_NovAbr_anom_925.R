library(HiddenMarkov)
library(fitdistrplus)


###############################   Nov_Abr   ################################

data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_PP_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd = c(data$V1)

fl <- fitdist(spd, "logis")

"2 Estados"
Pi    <- rbind(c(0.7, 0.3), c(0.4, 0.6))
delta <- c(1,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states2 <- Viterbi(x)

"3 Estados"
Pi    <- rbind(c(0.7, 0.2, 0.1), 
               c(0.3, 0.6, 0.1), 
               c(0.2, 0.2, 0.6))
delta <- c(1,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states3 <- Viterbi(x)

"4 Estados"
Pi    <- rbind(c(0.8, 0.1, 0.05, 0.05), 
               c(0.1, 0.6, 0.25, 0.05), 
               c(0.1, 0.15, 0.5, 0.25), 
               c(0.2, 0.05, 0.2, 0.55))
delta <- c(1,0,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale =c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states4 <- Viterbi(x)



"5 Estados"
Pi    <- rbind(c(0.7,  0.1,  0.05, 0.05, 0.1), 
               c(0.05, 0.6,  0.25, 0.05, 0.05), 
               c(0.1,  0.1,  0.5,  0.25, 0.05), 
               c(0.2,  0.05, 0.15, 0.55, 0.05), 
               c(0.05, 0.3,  0.1,  0.05, 0.5))

delta <- c(1,0,0,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states5 <- Viterbi(x)



"6 Estados"
Pi    <- rbind(c(0.65, 0.1, 0.05,  0.05, 0.07, 0.08), 
               c(0.02, 0.6,  0.2,  0.08, 0.05, 0.05), 
               c(0.1,  0.15, 0.5,  0.15, 0.05, 0.05), 
               c(0.07, 0.13, 0.05, 0.55, 0.15, 0.05), 
               c(0.05, 0.23, 0.1,  0.05, 0.5,  0.07), 
               c(0.15, 0.1,  0.03, 0.07, 0.15, 0.5))
delta <- c(1,0,0,0,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states6 <- Viterbi(x)


"7 Estados"
Pi    <- rbind(c(0.65, 0.1,  0.05, 0.02, 0.05, 0.07, 0.06), 
               c(0.02, 0.4,  0.2,  0.08, 0.05, 0.2,  0.05), 
               c(0.1,  0.1,  0.5,  0.15, 0.05, 0.05, 0.05), 
               c(0.03, 0.07, 0.13, 0.52, 0.05, 0.15, 0.05), 
               c(0.05, 0.2,  0.1,  0.03, 0.5,  0.05, 0.07), 
               c(0.15, 0.1,  0.1,  0.03, 0.07, 0.4,  0.15),
               c(0.17, 0.2,  0.05, 0.03, 0.07, 0.15, 0.33))

delta <- c(1,0,0,0,0,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states7 <- Viterbi(x)


"8 Estados"

Pi    <- rbind(c(0.45, 0.1,  0.07,  0.12, 0.03, 0.07, 0.07, 0.09), 
               c(0.02, 0.4,  0.15,  0.05, 0.08, 0.05, 0.2,  0.05), 
               c(0.1,  0.1,  0.5,   0.07, 0.08, 0.05, 0.05, 0.05), 
               c(0.02, 0.03, 0.07,  0.5,  0.13, 0.05, 0.15, 0.05), 
               c(0.05, 0.2,  0.1,   0.03, 0.3,  0.05, 0.07, 0.2), 
               c(0.13, 0.1,  0.05,  0.1,  0.03, 0.4,  0.07, 0.12),
               c(0.1,  0.1,  0.1,   0.03, 0.07, 0.15, 0.33, 0.12),
               c(0.13, 0.07, 0.1,   0.02, 0.03, 0.15, 0.08, 0.42))

delta <- c(1,0,0,0,0,0,0,0)
pm    <- list(location=c(fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1], fl$estimate[1]), 
              scale=c(fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2], fl$estimate[2]))

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=10000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states8 <- Viterbi(x)


"Datframe"

DF <- data.frame("states2" = states2, "states3" = states3, "states4" = states4, "states5" = states5, "states6" = states6, 'states7' = states7, 'states8' = states8)
write.csv(DF, 
          file = "/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/26_expo_2018/States_PP_NovAbr_anom_925.csv")

