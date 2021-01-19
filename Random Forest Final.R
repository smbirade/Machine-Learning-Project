#Random Forests Project 
library(dplyr)
#install.packages('randomForest')
library(randomForest)
library('ROCR')
library('pROC')

load("/Users/sierrabirade/Downloads/MLProjectData21.RData")
#Setting the seed
set.seed(1234) 
#Downsampling
train_subset <- sample_n(train, 125000) 
#Removing the other targets
train_subset <- subset(train_subset, select = -c(cont.target, binary.target2)) 

# Making the binary target a factor
train_subset$binary.target1 <- as.character(train_subset$binary.target1)
train_subset$binary.target1 <- as.factor(train_subset$binary.target1)

?randomForest()

## 1st Attempt 
# Random forest with 20 trees and 45 splits
rf=randomForest(binary.target1 ~ ., data=train_subset, ntree=20,mtry=45, type='class')
#Variable importance
rf$importance 
#Plot by variable importance
varImpPlot(rf) 
#Overall results 
rf 

# Test on validation 
#valid <- subset(valid, select = -c(cont.target, binary.target2)) #remove other targets
#Predicting on validation 
predicted <- predict(rf, valid[,-128]) 
#Confusion matrix
table(predicted, valid$binary.target1) 

#Creating the predictions with repsonses 
predictions <- as.numeric(predict(rf, valid[,-128], type="response"))
pred <- prediction(predictions, valid$binary.target1)
#Creating the ROC curve
roc1<- performance(pred, measure='tpr', x.measure='fpr')
#Plotting the ROC cruve
plot(roc1) 
#Getting the AUC
auc <- performance(pred, measure='auc')
auc 


## Attempt 2
#121 trees and 20 splits 
rf2=randomForest(binary.target1 ~ ., data=train_subset, ntree=121,mtry=20, type='class')
#Variable importance
rf2$importance 
#Plot by variable importance
varImpPlot(rf2) 
#Overall results 
rf2

# Testing on validation 
#valid <- subset(valid, select = -c(cont.target, binary.target2)) #remove other targets
#Predicting on validation 
predicted2 <- predict(rf2, valid[,-128]) 
#Confusion matrix
table(predicted2, valid$binary.target1) 

#Predicting the responses 
predictions2 <- as.numeric(predict(rf2, valid[,-128], type="response"))
pred2 <- prediction(predictions2, valid$binary.target1)
#Creating the ROC curve
perf2 <- performance(pred2, measure='tpr', x.measure='fpr') 
#Plotting the ROC curve
plot(perf2) 

#Calculating the AUC
auc <- performance(pred2, measure='auc')
auc 

