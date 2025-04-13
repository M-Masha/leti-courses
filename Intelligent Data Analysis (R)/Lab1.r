# Подготавливаем данные

library(XML)
website <- "http://www.pogodaiklimat.ru/history/23849.htm"
yearsData <- readHTMLTable(website, which = 1)
weatherData <- readHTMLTable(website, which = 2)

monthsList <- c('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Year Average')

yearsListM <- matrix(0, 100, 1)
for (i in 39:138) {
  yearsListM[i - 38, 1] <- as.numeric(yearsData[i, 1])
}
yearsList <- c(as.character(yearsListM))

task3names <- c('Volume', 'Min', 'Max', '1st Quartile', '3rd Quartile', 'IQR', 'Median', 'Mean', 'Standard Error', 'Standard Deviation', 'Coef of Var', 'Coef of Asym')


datasetM <- matrix(0, 100, 13)
for (i in 39:138) {
  for (j in 1:13) {
    datasetM[i - 38, j] <- as.numeric(weatherData[i, j])
  }
}
dataset <- as.data.frame(datasetM)
colnames(dataset) <- monthsList
rownames(dataset) <- yearsList

# Задание 3

task3M <- matrix(0, 12, 13)
task3 <- as.data.frame(task3M)
colnames(task3) <- monthsList
rownames(task3) <- task3names


volume = length(dataset[, 1])
for (i in 1:13) {
  task3[1, i] <- as.numeric(length(dataset[, i]))
}

minT <- sapply(dataset, min)
task3[2, ] <- minT

maxT <- sapply(dataset, max)
task3[3, ] <- maxT

fstQuart <- sapply(dataset, quantile)[2, ]
task3[4, ] <- fstQuart

trdQuart <- sapply(dataset, quantile)[4, ]
task3[5, ] <- trdQuart

IQR <- trdQuart - fstQuart
task3[6, ] <- IQR

medianT <- sapply(dataset, median)
task3[7, ] <- medianT

meanT <- sapply(dataset, mean)
task3[8, ] <- meanT

standardD <- sapply(dataset, sd)
standardE <- standardD / sqrt(volume)
task3[9, ] <- standardE

task3[10, ] <- standardD

coefvar <- abs(standardD / meanT * 100)
task3[11, ] <- coefvar

distMoment3 <- integer(13)
coefasym <- integer(13)
for (i in 1:13) {
  distMoment3[i] <- sum((dataset[, i] - meanT[i]) ^ 3) / volume
  coefasym[i] <- (distMoment3[i]) / (standardD[i] ^ 3)
}
task3[12, ] <- coefasym


for (i in 1:12) {
  for (j in 1:13) {
    task3[i, j] = round(task3[i, j], digits = 3)
  }
}

# Задание 4.1

for (i in 1:13) {
  boxplot(dataset[, i], main=paste("Boxplot for", monthsList[i]))
}

boxplot(dataset[, -13], main="Boxplot for 12 months")

# Задание 4.2

outliersM <- matrix(0, 100, 13)
colnames(outliersM) <- monthsList
rownames(outliersM) <- yearsList

for (i in 1:100) {
  for (j in 1:13) {
    if (dataset[i, j] < fstQuart[j] - (IQR[j]*1.5)) {
      outliersM[i, j] <- dataset[i, j]
    }
    if (dataset[i, j] > trdQuart[j] + (IQR[j]*1.5)) {
      outliersM[i, j] <- dataset[i, j]
    }
  }
}

outliers <- as.data.frame(outliersM)
for (i in 100:1) {
  if (sum(abs(outliersM[i, ])) == 0) {
    outliers <- outliers[-i, ]
  }
}

# Задание 5

for (i in 1:13) {
  hist(dataset[, i], breaks = 9, main=paste("Histogram for", monthsList[i]))
}

# Задание 6.1.1
task6names <- c("Median - Mean", "-3 Sigma", "+3 Sigma", "Min", "Max", "IQR to SD")

task6M <- matrix(0, 6, 13)
colnames(task6M) <- monthsList
rownames(task6M) <- task6names
task6 <- as.data.frame(task6M)

task6[1, ] <- task3[7, ] - task3[8, ]

task6[2, ] <- meanT - standardD * 3

task6[3, ] <- meanT + standardD * 3

task6[4, ] <- minT

task6[5, ] <- maxT

task6[6, ] <- IQR - standardD * 1.33

for (i in 1:6) {
for (j in 1:13) {
task6[i, j] = round(task6[i, j], digits = 3)
}
}


# Задание 6.1.3

for (i in 1:13) {
  qqnorm(dataset[, i], main=paste("Q-Q Plot for", monthsList[i]))
  qqline(dataset[, i])
}

# Задание 6.1.4

for (i in 1:13) {
  temp = shapiro.test(dataset[, i])[2]
  temp = paste("Shapiro-Wilk test for", monthsList[i], round(as.numeric(temp), digits = 3))
  print(temp)
}