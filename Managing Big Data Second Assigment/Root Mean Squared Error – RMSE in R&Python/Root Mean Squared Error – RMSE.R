calculateRMSE<-function(predictedValues, actualValues){
  err<- sqrt( mean((actualValues - predictedValues)^2)  )
  return( err )
}

kFoldCrossValidation<-function(data, frml, k){
  dataset<-data[sample(nrow(data)),]
  folds <- cut(seq(1,nrow(dataset)), breaks=k, labels=FALSE)
  RMSE<-vector()
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- dataset[testIndexes, ]
    trainData <- dataset[-testIndexes, ]
    candidate.linear.model<-lm( frml, data = trainData)
    predicted<-predict(candidate.linear.model, testData)
    error<-calculateRMSE(predicted, testData[, "area"])
    RMSE<-c(RMSE, error)
  }
  return( mean(RMSE) )
}

forestfires<-read.csv("C:/metaptyxiako/diaxeirhsh megalwn dedomenwn/Assigment2/ergasia/ergasia/forestfires.csv", sep=",", header=T, stringsAsFactors = F)
forestfires<-na.omit(forestfires)

predictionModel<-"area ~ temp+wind+rain"

  modelErr<-kFoldCrossValidation(forestfires, as.formula(predictionModel), 10)
  modelMeanRMSE<-c(predictionModel, modelErr)
  print( sprintf("Linear regression model [%s]: prediction error [%f]", predictionModel, modelErr ) )

##### task ii ####
subfore = subset(forestfires, forestfires$area<3.2 )
modelErr<-kFoldCrossValidation(subfore, as.formula(predictionModel), 10)

modelMeanRMSE<-c(modelMeanRMSE, modelErr)
print( sprintf("Linear regression model [%s]: prediction error [%f]", predictionModel, modelErr ) )


