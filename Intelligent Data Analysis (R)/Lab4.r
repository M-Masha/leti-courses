# 1.Подготавливаем данные

library(XML)
library(lattice)
library(ggplot2)
library(car)
library(rstatix)

website <- "http://www.pogodaiklimat.ru/history/23849.htm"
yearsData <- readHTMLTable(website, which = 1)
weatherData <- readHTMLTable(website, which = 2)

monthsList <- c('1', '2', '3', '4', '5', '6', '7', '8', '9')

yearsListM <- matrix(0, 10, 1)
for (i in 129:138) {
  yearsListM[i - 128, 1] <- as.numeric(yearsData[i, 1])
}
yearsList <- c(as.character(yearsListM))

datasetM <- matrix(0, 10, 9)
for (i in 129:138) {
  for (j in 1:9) {
    datasetM[i - 128, j] <- as.numeric(weatherData[i, j])
  }
}
dataset <- as.data.frame(datasetM)
colnames(dataset) <- monthsList
rownames(dataset) <- yearsList

dpmatrixM <- matrix(0, 90, 2)
for (i in 1:10) {
  dpmatrixM[i, 1] <- c("1")
  dpmatrixM[i, 2] <- as.numeric(datasetM[i, 1])
  
  dpmatrixM[i + 10, 1] <- c("2")
  dpmatrixM[i + 10, 2] <- as.numeric(datasetM[i, 2])
  
  dpmatrixM[i + 20, 1] <- c("3")
  dpmatrixM[i + 20, 2] <- as.numeric(datasetM[i, 3])
  
  dpmatrixM[i + 30, 1] <- c("4")
  dpmatrixM[i + 30, 2] <- as.numeric(datasetM[i, 4])
  
  dpmatrixM[i + 40, 1] <- c("5")
  dpmatrixM[i + 40, 2] <- as.numeric(datasetM[i, 5])
  
  dpmatrixM[i + 50, 1] <- c("6")
  dpmatrixM[i + 50, 2] <- as.numeric(datasetM[i, 6])
  
  dpmatrixM[i + 60, 1] <- c("7")
  dpmatrixM[i + 60, 2] <- as.numeric(datasetM[i, 7])
  
  dpmatrixM[i + 70, 1] <- c("8")
  dpmatrixM[i + 70, 2] <- as.numeric(datasetM[i, 8])
  
  dpmatrixM[i + 80, 1] <- c("9")
  dpmatrixM[i + 80, 2] <- as.numeric(datasetM[i, 9])
}

dpmatrix <- as.data.frame(dpmatrixM)
colnames(dpmatrix) <- c("month", "tmps")

dpmatrix[, 'tmps'] <- as.numeric(dpmatrix[, 'tmps'])
dpmatrix[, 'month'] <- as.factor(dpmatrix[, 'month'])

#2.Диаграммы разброса

dotplot(tmps ~ month, data = dpmatrix)

ggplot(dpmatrix) +
  aes(x = month, y = tmps, color = month) +
  geom_jitter() +
  theme(legend.position = "none")

model <- lm(dpmatrix$tmps ~ dpmatrix$month, data=dpmatrix)


# 3.Коэффициент корреляции

dpmatrix[] <- lapply(dpmatrix, function(x) as.numeric(x))
corr = cor(dpmatrix$month, dpmatrix$tmps)

# Коэффициент Стьюдента

t <- corr/(sqrt((1-corr^2)/88))


# 4. Регрессионные коэффициенты наклона и сдвига

ssxy <- function(x, y) {
  return (sum((x - mean(x))*(y-mean(y))))
}

ssx <- function(x) {
  return (sum((x - mean(x))^2))
}

b1 <- ssxy(dpmatrix$month, dpmatrix$tmps)/ssx(dpmatrix$month)

b0 <- mean(dpmatrix$tmps) - b1*mean(dpmatrix$month)

coefficients(model)


# SST, SSR, SSE
sst <- sum((dpmatrix$tmps - mean(dpmatrix$tmps))^2)

ssr <- b0*sum(dpmatrix$tmps) + b1*sum(dpmatrix$month*dpmatrix$tmps) -
  ((sum(dpmatrix$tmps))^2)/(nrow(dpmatrix))

sse <- sum(dpmatrix$tmps^2) - b0*sum(dpmatrix$tmps) -
  b1*sum(dpmatrix$month*dpmatrix$tmps)

# Среднеквадратическая ошибка оценки

syx <- sqrt(sse/(nrow(dpmatrix)-2))


# Коэффициент детерминации

r2 <- ssr/sst


var_frame <- data.frame(SST = sst, SSR = ssr, SSE = sse, S_yx = syx,
                        r2 = r2)
round(var_frame, digits = 2)

# 6.90%-доверительный интервал для коэффициента наклона

sb1 <- syx/sqrt(ssx(dpmatrix$month))
l <- b1 -  1.6623540*sb1
r <- b1 +  1.6623540*sb1

confint_frame <- data.frame(left = l, righ = r)
round(confint_frame, digits = 4)


# 7.Уравнение регрессии
Y <- b0 + b1*dpmatrix$month

# Линия регрессии на диаграмме разброса
ggplot(dpmatrix, aes(x = month, y = tmps)) +
  geom_point()+ geom_smooth(method = lm)


# 8.График зависимости остатков линейной модели от времени
ggplot(model, aes(x=fitted(model), y=residuals(model)))+
  geom_point() + geom_hline(yintercept = 0, linetype = "dashed" )

# 9.График квантиль-квантиль
car::qqPlot(residuals(model))


# 10.Применение критерия Дарбина-Уотсона
car::durbinWatsonTest(model)

