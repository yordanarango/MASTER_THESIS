library(fitdistrplus)
library(ggplot2)

#############################  PANAMA  #############################################

ch = "PN_ALT"
data <- read.csv(file=paste0("/home/yordan/YORDAN/UNAL/TESIS_MAESTRIA/18_expo_2018/datos_", ch, "_EneDic_1979_1999.csv")
                 , header=FALSE, sep=",")

spd  <- c(data$V1)

"Ajustes"
dgm <- fitdist(spd, "gamma")         # gamma
dln <- fitdist(spd, "lnorm")         # lognormal
dlg <- fitdist(spd, "logis")         # logistica
dwb <- fitdist(spd, "weibull")       # weibull

"Graficas todos"
listdis     <- list(dgm, dln, dlg, dwb)
plot_legend <- c("Gamma", "Lognormal", "Logistic", "Weibull")

denscomp(listdis, legendtext = plot_legend, main = paste0(ch," - Speed PDF (79-99)"),  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue","red","goldenrod", "dimgray"), 
         fitlty=c(1,1,1,1), ylim=c(0,0.24), plotstyle = "ggplot",
         breaks=20) +
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
 
   geom_line(size=0.6)

"INDIVIDUALES"
"parametros"
ymax = 0.24

listdis     <- list(dgm)
plot_legend <- c("Gamma")

denscomp(listdis, legendtext = plot_legend, main = paste0(ch," - Speed PDF (79-99)"),  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("blue"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)



listdis     <- list(dln)
plot_legend <- c("Lognormal")

denscomp(listdis, legendtext = plot_legend, main = paste0(ch," - Speed PDF (79-99)"),  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("red"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)



listdis     <- list(dlg)
plot_legend <- c("Logistic")

denscomp(listdis, legendtext = plot_legend, main = paste0(ch," - Speed PDF (79-99)"),  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("goldenrod"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)



listdis     <- list(dwb)
plot_legend <- c("Weibull")

denscomp(listdis, legendtext = plot_legend, main = paste0(ch," - Speed PDF (79-99)"),  
         xlab = "Speed [m/s]", ylab = "PDF", 
         fitcol=c("dimgray"), 
         fitlty=c(1), ylim=c(0, ymax), plotstyle = "ggplot",
         breaks=20) + 
  
  theme_bw() + 
  
  theme(legend.text=element_text(size=13), legend.position = c(0.85, 0.8), 
        plot.title = element_text(hjust = 0.5))+
  
  geom_line(size=0.7)
