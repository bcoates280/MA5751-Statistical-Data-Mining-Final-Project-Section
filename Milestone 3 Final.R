#Load Required Libraries
library(caret)
library(corrplot) 
library(pls)
library(mlbench)
library(e1071)
library(pROC)
library(MASS)
library(kernlab)
library(earth)
library(klaR)
library(svmpath)

#Load Data Set
cancer <- read.csv("cancer_reg.csv")

#Remove un-wanted predictors
GeographyIndex <- grep("Geography", colnames(cancer))
binnedIncIndex <- grep("binnedInc", colnames(cancer))
AvgDeathsIndex <- grep("avgDeathsPerYear",colnames(cancer))
AvgCountIndex <- grep("avgAnnCount", colnames(cancer))
incidenceIndex <- grep("incidenceRate", colnames(cancer))
cancer <- cancer[, -c(GeographyIndex, binnedIncIndex, AvgCountIndex, AvgDeathsIndex, incidenceIndex)]

#Check for NA's in response variable
TARGET_deathRateNAs <- is.na(cancer$TARGET_deathRate)
NAsTrue <- grep("TRUE", TARGET_deathRateNAs)
NAsTrue

#Check for Na's in predictor variables
NAs <- is.na(cancer)
NAsTrue <- grep("TRUE", NAs)
length(NAsTrue)

#Remove Predictors with large NA counts
PctSomeCol18_24Index <- grep("PctSomeCol18_24", colnames(cancer))          
PctEmployed16_OverIndex <- grep("PctEmployed16_Over", colnames(cancer))               
PctPrivateCoverageAloneIndex <- grep("PctPrivateCoverageAlone", colnames(cancer))
cancer <- cancer[, -c(PctSomeCol18_24Index, PctEmployed16_OverIndex,
                      PctPrivateCoverageAloneIndex)]

# Convert Response Variable into binary
colnames(cancer)
death_mean <- 144.1
breaks  <- c(-Inf, death_mean, Inf) 
labels  <- c("L", "H")  #0 - lower, 1-higher
binary_deathRate  <- cut(x=cancer$TARGET_deathRate, breaks=breaks, labels=labels, right=F)
cancer <- data.frame(cancer, binary_deathRate)
head(cancer$binary_deathRate) # confirm dataframe includes new variable

# drop original response variable
cancer = subset(cancer, select = -c(TARGET_deathRate) )
head(cancer$binary_deathRate)

#Subset data before transforming, dont want to transform a binary variable
cancer_pred = subset(cancer, select = -c(binary_deathRate) ) + 0.5
cancer_resp = subset(cancer, select = c(binary_deathRate) )

#Check skewness of predictor variables
skew <- apply(cancer_pred, 2, skewness)
skew

#Perform centering, scaling, and Box Cox transformation on predictor variables
trans <- preProcess(cancer_pred, method = c("center", "scale", "BoxCox"))
trans

#Apply the transformation to the data set
cancerTrans <- predict(trans, cancer_pred)

#Check skewness of predictor variables after transformation
skew.transform <- apply(cancerTrans, 2, skewness)
skew.transform

#Check for near zero variance predictor variables
nearZeroVar(cancerTrans) # None

#Construct a correlation plot of data
par(mfrow = c(1,1))
corrplot::corrplot(cor(cancerTrans), order = "hclust")

#Identify highly correlated predictor variables
cor90 <- findCorrelation(cor(cancerTrans), cutoff = 0.90)
cor90

# view which column is high correlation
colnames(cancerTrans)[cor90]

#Remove highly correlated predictors
cancerTrans <- cancerTrans[-c(cor90)]

# combine predictor and response dataframes again
transformed_df <- data.frame(cancer_resp, cancerTrans)
transformed_df$binary_deathRate <- as.factor(transformed_df$binary_deathRate)
head(transformed_df, 2)

#Conduct data splitting
set.seed(0)
trainIndex <- sample(1:nrow(transformed_df), nrow(transformed_df)*0.8, replace = FALSE)
length(trainIndex)

#Define training set and testing set
cancerTransTrain <- transformed_df[trainIndex,]
cancerTransTest <- transformed_df[-c(trainIndex),]
head(cancerTransTest)

#split output and predictors
TARGET_deathRateIdx <- grep("binary_deathRate", colnames(cancerTransTrain))
trainCanP <- cancerTransTrain[, -TARGET_deathRateIdx]
trainCanR <- cancerTransTrain[, TARGET_deathRateIdx]
testCanP <- cancerTransTest[, -TARGET_deathRateIdx]
testCanR <- cancerTransTest[, TARGET_deathRateIdx]

head(cancerTransTrain)

#Set cross-validation control
ctrl <- trainControl(method = "LGOCV",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

# logistic regression
# train the model on training set
lrfit <- train(binary_deathRate ~.,
               data = cancerTransTrain,
               trControl = ctrl,
               method = "glm",
               family=binomial(),
               metric = "ROC")

# print cv scores
summary(lrfit)

#confusion matrix - training data
confusionMatrix(data = lrfit$pred$pred,
                reference = lrfit$pred$obs)

#ROC curve - training data
trainROC.lr <- roc(response = lrfit$pred$obs,
                   predictor = lrfit$pred$H,
                   levels = rev(levels(lrfit$pred$obs)))
par(mfrow = c(1,1))
plot(trainROC.lr, legacy.axes = TRUE, main = "Training Data ROC Curve: Logistic Regression")
auc(trainROC.lr)


# predict on testing
lrPred <- predict(lrfit, testCanP)
summary(lrPred)
lrValues <- postResample(pred = lrPred, obs = testCanR)
lrValues

confusionMatrix(data = as.factor(lrPred), 
                reference = testCanR)

# AUC
lrRoc <- roc(response = lrfit$pred$obs,
             predictor = lrfit$pred$H,
             levels = rev(levels(lrfit$pred$obs)))
plot(lrRoc, legacy.axes = TRUE, main = "Logistic Regression")
lrAUC <- auc(lrRoc)
lrAUC

# Naive Bayes
set.seed(1000)
nbFit = train(binary_deathRate ~.,
              data = cancerTransTrain,
              trControl = ctrl,
              method = "nb", 
              metric = "ROC")
summary(nbFit)
par(mfrow = c(1,1))
plot(nbFit)

nbImpSim <- varImp(nbFit, scale = FALSE)
nbImpSim
plot(nbImpSim, top = 5, scales = list(y = list(cex = .95)))

# predict on testing
nbPred <- predict(nbFit, testCanP)
summary(nbPred)
nbValues <- postResample(pred = nbPred, obs = testCanR)
nbValues

confusionMatrix(data = as.factor(nbPred), 
    reference = testCanR)

# AUC
nbRoc <- roc(response = nbFit$pred$obs,
             predictor = nbFit$pred$H,
             levels = rev(levels(nbFit$pred$obs)))
plot(nbRoc, legacy.axes = TRUE, main = "Naive Bayes")
nbAUC <- auc(nbRoc)
nbAUC

# LDA
set.seed(1000)
ldaFit <- train(x = trainCanP,
                y = trainCanR,
                method = "lda",
                preProc = c("center", "scale"),
                metric = "ROC",
                trControl = ctrl)
ldaFit

# predict on test set
ldaPred = predict(ldaFit, testCanP)
ldaValues <- postResample(pred = ldaPred, obs = testCanR) 
ldaValues

# test set
confusionMatrix(data = ldaPred,
                reference = testCanR)

ldaImpSim <- varImp(ldaFit, scale = FALSE)
ldaImpSim
plot(ldaImpSim, top = 5, scales = list(y = list(cex = .95)))

# AUC
ldaRoc <- roc(response = ldaFit$pred$obs,
              predictor = ldaFit$pred$H,
              levels = rev(levels(ldaFit$pred$obs)))
plot(ldaRoc, legacy.axes = TRUE, main = "Linear Discriminant Analysis")
ldaAUC <- auc(ldaRoc)
ldaAUC


# SVM 
set.seed(1000)
#Define svmgrid
sigmaRangeReduced <- sigest(as.matrix(trainCanP))
svmGrid <- expand.grid(.sigma = sigmaRangeReduced,
                       .C = 2^(seq(-4, 6)))

SVMtrain <- train(binary_deathRate ~.,
                  data = cancerTransTrain,
                  method = "svmRadial",
                  metric = "ROC",
                  tuneGrid = svmGrid,
                  trControl = ctrl)
SVMtrain
plot(SVMtrain)

plot(varImp(SVMtrain), 10, main="SVM-Importance")
SVMtrain$finalModel

#Predict on Test Data
testCanP$SVM.class <- predict(SVMtrain, testCanP)
predict.SVM <- predict(SVMtrain, testCanP, type="prob")
testCanP$SVMprob <- predict.SVM[,"H"]
testCanP$obs <- as.factor(testCanR)

#confusion matrix - testing data
confusionMatrix(data = testCanP$SVM.class,
                reference = testCanP$obs,
                positive = "H")

#ROC Curve - testing data
testROC.SVM <- roc(response = testCanP$obs,
                   predictor = testCanP$SVMprob,
                   levels = rev(levels(testCanP$obs)))
plot(testROC.SVM, legacy.axes = TRUE, main = "Testing Data ROC Curve: SVM")
auc(testROC.SVM)


#QDA
qdaFit <- train(x = trainCanP,
                y = trainCanR,
                method = "qda",
                preProc = c("center", "scale"),
                metric = "ROC",
                trControl = ctrl)
qdaFit

plot(varImp(qdaFit), 10, main="QDA-Importance")
qdaFit$finalModel

#Predict on Test Data
testCanP$QDA.class <- predict(qdaFit, testCanP)
predict.QDA <- predict(qdaFit, testCanP, type="prob")
testCanP$QDAprob <- predict.QDA[,"H"]
testCanP$obs <- as.factor(testCanR)

#confusion matrix - testing data
confusionMatrix(data = testCanP$QDA.class,
                reference = testCanP$obs,
                positive = "H")

#ROC Curve - testing data
testROC.QDA <- roc(response = testCanP$obs,
                   predictor = testCanP$QDAprob,
                   levels = rev(levels(testCanP$obs)))
plot(testROC.QDA, legacy.axes = TRUE, main = "Testing Data ROC Curve: QDA")
auc(testROC.QDA)
