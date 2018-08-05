library(HiddenMarkov)
library(fitdistrplus)
library(ncdf4)
library(ggplot2)

############################# TEHUANTEPEC ###############################################
data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_TT_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd  <- c(data$V1)
spd  <- spd/max(spd) * 0.9999999

"Ajustes"
dgm <- fitdist(spd, "gamma")               # gamma
dln <- fitdist(spd, "lnorm")               # lognormal
dlg <- fitdist(spd, "logis")               # logistica
dwb <- fitdist(spd, "weibull")             # weibull
dbt <- fitdist(spd, "beta", method="mge")  # beta: los valores deben ser dados entre 0 y 1

#"Godness of fit - Beta"
#Loglikelihood = sum(log(dbeta(spd, dbt$estimate[1], dbt$estimate[2])))
#AIC           = Loglikelihood*-2 + 4
#BIC           = Loglikelihood*-2 + 2*log(length(spd))

"Graficas todos"
png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_Anom_925.png")
listdis     <- list(dgm, dln, dlg, dbt)
plot_legend <- c("Gamma", "Lognormal", "Logistic", "Beta")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue","red","goldenrod", "black"), 
         fitlty=c(1,1,1,1), ylim=c(0,3.1), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5)) +
  
  geom_line(size=0.6)
dev.off()

"INDIVIDUALES"
"paramestros"
ymax = 3.1

png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_Gamma_Anom_925.png")
listdis     <- list(dgm)
plot_legend <- c("Gamma")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_LNormal_Anom_925.png")
listdis     <- list(dln)
plot_legend <- c("Lognormal")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("red"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_Logistic_Anom_925.png")
listdis     <- list(dlg)
plot_legend <- c("Logistic")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("goldenrod"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_Weibull_Anom_925.png")
listdis     <- list(dwb)
plot_legend <- c("Weibull")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("dimgray"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/TT_PDF_Beta_Anom_925.png")
listdis     <- list(dbt)
plot_legend <- c("Beta")

denscomp(listdis, legendtext = plot_legend, main = "TT - Speed PDF (Nov-Abr)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("black"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()

############################# PAPAGAYO ###############################################
data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_PP_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd  <- c(data$V1)
spd  <- spd/max(spd) * 0.9999999

"Ajustes"
dgm <- fitdist(spd, "gamma")         # gamma
dln <- fitdist(spd, "lnorm")         # lognormal
dlg <- fitdist(spd, "logis")         # logistica
dwb <- fitdist(spd, "weibull")       # weibull
dbt <- fitdist(spd, "beta", method="mge")  # beta: los valores deben ser dados entre 0 y 1

#"Godness of fit - Beta"
#Loglikelihood = sum(log(dbeta(spd[-which(spd == 1)], dbt$estimate[1], dbt$estimate[2])))
#AIC           = Loglikelihood*-2 + 4
#BIC           = Loglikelihood*-2 + 2*log(length(spd))

"Graficas todos"
png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_Anom_925.png")
listdis     <- list(dgm, dln, dlg, dbt)
plot_legend <- c("Gamma", "Lognormal", "Logistic", "Beta" )

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue","red","goldenrod", "black"), 
         fitlty=c(1,1,1,1), , ylim=c(0,2.8), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.6)
dev.off()

"INDIVIDUALES"
"parametros"
ymax = 2.8

png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_Gamma_Anom_925.png")
listdis     <- list(dgm)
plot_legend <- c("Gamma")

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_LNormal_Anom_925.png")
listdis     <- list(dln)
plot_legend <- c("Lognormal")

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("red"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_Logistic_Anom_925.png")
listdis     <- list(dlg)
plot_legend <- c("Logistic")

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("goldenrod"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_Weibull_Anom_925.png")
listdis     <- list(dwb)
plot_legend <- c("Weibull")

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("dimgray"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PP_PDF_Beta_Anom_925.png")
listdis     <- list(dbt)
plot_legend <- c("Beta")

denscomp(listdis, legendtext = plot_legend, main = "PP - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("black"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()

#############################  PANAMA  #############################################
data <- read.csv(file="/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/datos_PN_NovAbr_anom_925.csv"
                 , header=FALSE, sep=",")

spd  <- c(data$V1)
spd  <- spd/max(spd) * 0.9999999

"Ajustes"
dgm <- fitdist(spd, "gamma")         # gamma
dln <- fitdist(spd, "lnorm")         # lognormal
dlg <- fitdist(spd, "logis")         # logistica
dwb <- fitdist(spd, "weibull")       # weibull
dbt <- fitdist(spd, "beta", method="mge")  # beta: los valores deben ser dados entre 0 y 1

#"Godness of fit - Beta"
#Loglikelihood = sum(log(dbeta(spd[-which(spd == 1)], dbt$estimate[1], dbt$estimate[2])))
#AIC           = Loglikelihood*-2 + 4
#BIC           = Loglikelihood*-2 + 2*log(length(spd))

"Graficas todos"
png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_Anom_925.png")
listdis     <- list(dgm, dln, dlg, dbt)
plot_legend <- c("Gamma", "Lognormal", "Logistic", "Beta")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue","red","goldenrod", "black"), 
         fitlty=c(1,1,1,1), ylim=c(0,2.8), plotstyle = "ggplot",
         breaks=20) +
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
 
   geom_line(size=0.6)
dev.off()

"INDIVIDUALES"
"parametros"
ymax = 2.8

png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_Gamma_Anom_925.png")
listdis     <- list(dgm)
plot_legend <- c("Gamma")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_LNormal_Anom_925.png")
listdis     <- list(dln)
plot_legend <- c("Lognormal")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("red"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_Logistic_Anom_925.png")
listdis     <- list(dlg)
plot_legend <- c("Logistic")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("goldenrod"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_Weibull_Anom_925.png")
listdis     <- list(dwb)
plot_legend <- c("Weibull")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("dimgray"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()


png("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/25_expo_2018/PN_PDF_Beta_Anom_925.png")
listdis     <- list(dbt)
plot_legend <- c("Beta")

denscomp(listdis, legendtext = plot_legend, main = "PN - Speed PDF (Nov-Mar)",  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("black"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
dev.off()
