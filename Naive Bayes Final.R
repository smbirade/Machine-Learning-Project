#######################################################
#######################################################
########Creating Train and Validation Datasets#########
#######################################################
#######################################################
test2=train
b_valid2=valid
test2=test2[-c(7,45,128,130)]
b_valid2=b_valid2[-c(7,45,128,130)]
min(test2)
#######################################################
#######################################################
###########Trying Different Standardization############
#######################################################
#######################################################
#normal=normalize(test2,method="standardize")
#hist(normal[,7])
#normal2=normalize(test2,method="scale")
#hist(normal2[,1])
#normal3=normalize(test2,method="range")
#hist(normal3[,7])
#normal4=normalize(test2,method="center")
#hist(normal4[,7])
#######################################################
mlscale = apply(test2[,c(1:125)], 2, function(x)(x-min(x))/(max(x)-min(x)))
#This is the best standardization
#######################################################
#######################################################
#####Using PCA to hopefully make the data normal#######
#######################################################
#######################################################
cov.pca = prcomp(mlscale, scale=T)
pr.var=(cov.pca$sdev)^2
pve=pr.var/sum(pr.var)
#Choosing the number of principal components
plot(pve,xlab="PC", ylim=c(0,1),type="b")
plot(cov.pca$sdev^2, main = 'Covariance PCA',ylab='Eigenvalue')
plot(cov.pca$x)
#30 seems like the best choice based on the elbow plot
#######################################################
#######################################################
#####Creating new dataframe from pc, and target #######
#######################################################
#######################################################
pca=data.frame(cov.pca$x)
new=data.frame(pca[,1:30],train$binary.target1)
#Sampling the data
#train_samp=sample(c(T,F),nrow(new), replace=T, p=c(0.75,0.25))
#Setting the target to a factor
new$train.binary.target1=as.factor(new$train.binary.target1)
target=data.frame(new$train.binary.target1)

#######################################################
#######################################################
#########Modeling using a Naive Bayse ################
#######################################################
#######################################################
#Sample Model
#model = naiveBayes(x=new[train_samp,1:30], y=target[train_samp,])
#Model using all the data
model6 = naiveBayes(x=new[,1:30], y=new$train.binary.target1, laplace = 0.01)

#########################################################################
#########################################################################
##On the training data using PCA to hopefully make the data normal#######
#########################################################################
#########################################################################
mlscale2 = apply(b_valid2[,c(1:125)], 2, function(x)(x-min(x))/(max(x)-min(x)))
cov.pca2 = prcomp(mlscale2, scale=T)
pr.var2=(cov.pca2$sdev)^2
pve2=pr.var2/sum(pr.var2)
#Choosing the number of principal components
plot(pve2,xlab="PC", ylim=c(0,1),type="b")
plot(cov.pca2$sdev^2, main = 'Covariance PCA',ylab='Eigenvalue')
#30 seems like the best choice based on the elbow plot


########################################################################
########################################################################
#####Validation Data: Creating new dataframe from pc, and target #######
########################################################################
########################################################################
pca2=data.frame(cov.pca2$x)
new2=data.frame(pca2[,1:30],valid$binary.target1)
#Setting the target to a factor
new2$valid.binary.target1=as.factor(new2$valid.binary.target1)
target2=data.frame(new2$valid.binary.target1)

#######################################################
#######################################################
#########Predicting using a Naive Bayse #############
#######################################################
#######################################################
#Sample Predictions
#pred=predict(model,new[!train_samp,1:30], type="class")
#predications using the actual validation
pred6 = predict(model6,new2[,1:30], type="class")

#######################################################
#######################################################
######Cross Tables of Predictions Using NB ############
#######################################################
#######################################################
#Sample crosstable
#CrossTable(pred,target[!train_samp,], prop.chisq=F, prop.t = F, dnn = c('Predicted','Actual'))
#CrossTable on all and actual data
CrossTable(x=pred6,y=new2$valid.binary.target1, prop.chisq=F, prop.t = F, dnn = c('Predicted','Actual'))
#Misclassification rate
1-mean(pred6==new2$valid.binary.target1)

#######################################################
#######################################################
#########ROC Cureva and AUC, Using Probs ##############
#######################################################
#######################################################
pred7 = predict(model6,new2[,1:30], type="raw")
rf.roc4=roc(new2$valid.binary.target1,as.numeric(pred7[,2]))
plot(rf.roc4)
auc(rf.roc4)
#AUC is .6667

#######################################################
#######################################################
####################Lift ##############################
#######################################################
#######################################################
??lift
plotLift(as.numeric(pred7[,2]), new2$valid.binary.target1)
TopDecileLift(as.numeric(pred7[,2]), new2$valid.binary.target1)

table(test2$binary.target1)


