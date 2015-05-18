#### initial setup
rm(list = ls())
getwd()
setwd('/Users/hawooksong/Desktop/insult_detection')



#### load libraries
library(SnowballC)
library(tm)
library(stringr)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)



#### load data
missTypes <- c('', ' ')
train <- read.csv('./data/train.csv', na.strings=missTypes, stringsAsFactors=FALSE)
#finalTest <- read.csv('./data/final_test_with_solutions.csv', na.strings=missTypes, stringsAsFactors=FALSE)



#### load custom functions
source('./code/functions.R')



#### quick glance at the data
dim(train)
str(train)
table(train$Insult)



#### preprocessing
## lowercase column names
colnames(train) <- tolower(colnames(train))

## rename "date" column to "datetime"
colnames(train)[colnames(train)=='date'] <- 'datetime'

## fix the date column
train$datetime <- strptime(train$datetime, format='%Y%m%d%H%M%S')
train$datetime <- as.POSIXct(train$datetime)

trainBackup <- train

#### feature extraction
train <- addDateTimeFeaturesToDF(df=train, datetimeVar='datetime')
train <- addTextFeaturesToDF(df=train, textBodyVar='comment')
train <- replaceTextBodyWithSparseWordFreqMtrx(df=train, textBodyVar='comment', minWordAppPerc=0.02) 
train <- 


#### separate the train dataset to further split into "train" and "test"
PretrainCV <- splitToTrainAndCV(df, outcomeVar='insult', randNumSeed=123, splitRatio=0.75)
pretrain <- PretrainCV$train
cv <- PretrainCV$cv

dim(pretrain)
dim(cv)




#### establishing naive model base
table(pretrain$insult)
sum(pretrain$insult==0) / length(pretrain$insult)



#### logistic regression

## first attempt
# building a base model
logisModel1 <- glm(insult ~ . - insult - datetime - hour, data=pretrain, family=binomial)
summary(logisModel1)

# using step function to pick best variables and eliminate unimportant ones
logisModel2 <- step(logisModel1, direction='both')
summary(logisModel2)

## second attempt
logisModel3 <- glm(insult ~ . - date - hour, data=pretrain, family=binomial)
logisModel4 <- step(logisModel3, direction='both')

##  third attempt
# building models
pretrain$date <- pretrain$hour <- NULL
logisModel5 <- glm(insult ~ ., data=pretrain, family=binomial)
logisModel6 <- step(logisModel5, direction='both')
summary(logisModel6)

# prediction probability on pseudo-test data
probTestLogis <- predict(logisModel6, newdata=cv, type='response')
predTestLogis <- ifelse(probTestLogis >= 0.5, 1, 0)

# accuracy testing
confMtrxLogis <- table(predTestLogis, cv$insult)
calcAcc(confMtrxLogis)



#### decision tree

# building model
cartModel <- rpart(insult ~ ., data=pretrain, method='class')

# plot decision tree
prp(cartModel)

# prediction probabily on pseudo-test data
probTestCART <- predict(cartModel, newdata=cv, type='prob')[ , 2]
predTestCART <- ifelse(probTestCART >= 0.5, 1, 0)

# accuracy testing
confMtrxCART <- table(predTestCART, cv$insult)
calcAcc(confMtrxCART)


#### random forest

## convert  dependent variable to factor variable (required for random forest method)
pretrain$insult <- as.factor(pretrain$insult)
cv$insult <- as.factor(cv$insult)

## first attempt
# build a model
set.seed(123)
rfModel1 <- randomForest(insult ~ ., data=pretrain, nTree=100)

# plot important variables
varImpPlot(rfModel1, n.var = 25, 
           main = 'Importance of Variables',
           cex = 0.7)  # cex controls the size of label texts
# dev.copy(png, '../images/varImpPlot.png')
# dev.off()

# predictions
predTestRF1 <- predict(rfModel1, newdata=cv)
predTestRF1 <- as.integer(as.character(predTestRF1))

# accuracy
confMtrxRF1 <- table(predTestRF1, cv$insult)
calcAcc(confMtrxRF1)

## second attempt
# build a model
rfModel2 <- randomForest(insult ~ excCnt + capCnt + charCnt + atCnt + comCnt + cursCnt + can + fuck + get + like + right + your, 
                         data=pretrain, nTree=10)

# predictions
predTestRF2 <- predict(rfModel2, newdata=cv)
predTestRF2 <- as.integer(as.character(predTestRF2))

# accuracy
confMtrxRF2 <- table(predTestRF2, cv$insult)
calcAcc(confMtrxRF2)








