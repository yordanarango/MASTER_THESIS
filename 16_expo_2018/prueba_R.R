library(ncdf4)
library(HiddenMarkov)

archivo = nc_open('/home/yordan/YORDAN/UNAL/TRABAJO_DE_GRADO/DATOS_Y_CODIGOS/DATOS/U y V, 6 horas, 10 m, 1979-2016.nc')
time    = ncvar_get(archivo,"time")
U       = ncvar_get(archivo,"u10")[33, 9, seq(4,length(time),by=4)]
V       = ncvar_get(archivo,"v10")[33, 9, seq(4,length(time),by=4)]
spd     = sqrt(U*U+V*V)

Pi    <- rbind( c(0.6, 0.1, 0.3), c(0.2, 0.6, 0.2), c(0.3, 0.1, 0.6))
delta <- c(0,1,0)
pm = list(location=c( 2.1844435, 2.1844435, 2.1844435 ), scale=c(0.5588701, 0.5588701, 0.5588701))
#pm = list(rate=c( 0.5, 0.5, 0.5 ), shape=c(2, 2, 2))

x <- dthmm(NULL, Pi, delta, "gamma", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

x <- dthmm(NULL, Pi, delta, "lnorm", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

x <- dthmm(NULL, Pi, delta, "logis", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

x <- dthmm(NULL, Pi, delta, "weibull", pm , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE

y <- BaumWelch(x, bwcontrol(maxiter=1000, posdiff=FALSE))
x <- dthmm(NULL, y[["Pi"]], y[["delta"]], "logis", y[["pm"]] , discrete = FALSE)
x$x <- spd
x$nonstat <- FALSE
states <- Viterbi(x)
