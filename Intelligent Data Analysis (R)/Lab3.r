library(lattice)
library(ggplot2)
library(car)
library(rstatix)
library(XML)

webS <- "http://www.pogodaiklimat.ru/history/23849.htm"
webK <- "http://www.pogodaiklimat.ru/history/33345.htm"
webM <- "http://www.pogodaiklimat.ru/history/08222.htm"

yearsData <- readHTMLTable(webS, which = 1)
yearsListM <- matrix(0, 100, 1)
for (i in 39:138) {
  yearsListM[i - 38, 1] <- as.numeric(yearsData[i, 1])
}
yearsList <- c(as.character(yearsListM))

wthS <- readHTMLTable(webS, which = 2)
wthK <- readHTMLTable(webK, which = 2)
wthM <- readHTMLTable(webM, which = 2)

datasetM <- matrix(0, 100, 3)

for (k in 1:3) {
  if (k == 1) {
  for (i in 39:138) {
    datasetM[i - 38, k] <- as.numeric(wthS[i, 13])
  }
  }  
  if (k == 2) {
    for (i in 111:210) {
      datasetM[i - 110, k] <- as.numeric(wthK[i, 13])
    }
  }
  if (k == 3) {
    for (i in 83:182) {
      datasetM[i - 82, k] <- as.numeric(wthM[i, 13])
    }
  }
}

dataset <- as.data.frame(datasetM)
colnames(dataset) <- c("Surgut", "Kyiv", "Madrid")
rownames(dataset) <- yearsList


dpmatrixM <- matrix(0, 300, 2)
for (i in 1:100) {
  dpmatrixM[i, 1] <- c("Surgut")
  dpmatrixM[i, 2] <- as.numeric(datasetM[i, 1])
  
  dpmatrixM[i + 100, 1] <- c("Kyiv")
  dpmatrixM[i + 100, 2] <- as.numeric(datasetM[i, 2])
  
  dpmatrixM[i + 200, 1] <- c("Madrid")
  dpmatrixM[i + 200, 2] <- as.numeric(datasetM[i, 3])
}

dpmatrix <- as.data.frame(dpmatrixM)
colnames(dpmatrix) <- c("cities", "tmps")

dpmatrix[, 'tmps'] <- as.numeric(dpmatrix[, 'tmps'])
dpmatrix[, 'cities'] <- as.factor(dpmatrix[, 'cities'])

summary(dataset)

dotplot(tmps ~ cities, data = dpmatrix)

ggplot(dpmatrix) +
  aes(x = cities, y = tmps, color = cities) +
  geom_jitter() +
  theme(legend.position = "none")

res_aov <- aov(tmps ~ cities, data = dpmatrix)

hist(dataset$Surgut)
hist(dataset$Kyiv)
hist(dataset$Madrid)
hist(res_aov$residuals)

car::qqPlot(dataset$Surgut, id = FALSE)
car::qqPlot(dataset$Kyiv, id = FALSE)
car::qqPlot(dataset$Madrid, id = FALSE)
car::qqPlot(res_aov$residuals, id = FALSE)

shapiro.test(dataset$Surgut)
shapiro.test(dataset$Kyiv)
shapiro.test(dataset$Madrid)
shapiro.test(res_aov$residuals)

boxplot(tmps ~ cities, data = dpmatrix)

car::leveneTest(tmps ~ cities, data = dpmatrix)
car::leveneTest(tmps ~ cities, center = mean, data = dpmatrix)

oneway.test(tmps ~ cities, data = dpmatrix, var.equal = FALSE)

games_howell_test(dpmatrix, tmps ~ cities)
