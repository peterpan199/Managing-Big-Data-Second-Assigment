
calculateCost<-function(X, y, theta){
  m <- length(y)
  return( sum((X%*%theta- y)^2) / (2*m) )
} 

gradientDescent<-function(X, y, theta, alpha=0.01, numIters=90){
  m <- length(y)
  
  
  costHistory <- rep(0, numIters)

  for(i in 1:numIters){
    
   
    theta <- theta - alpha*(1/m)*(t(X)%*%(X%*%theta - y))
    
   
    costHistory[i]  <- calculateCost(X, y, theta)
    
  } 
  gdResults<-list("coefficients"=theta, "costs"=costHistory)
  return(gdResults)
} 



setwd("C:/metaptyxiako/diaxeirhsh megalwn dedomenwn/Managing Big Data Second Assigment/Topic 5")
HouseholdData<-read.csv("HouseholdData.csv", sep=",", header=T)

numObs<-nrow(HouseholdData)

revenue<- HouseholdData[, 2]

indVariables<- cbind( rep(1, numObs), HouseholdData$Income, HouseholdData$FamilySize ) 



initialThetas<-rep(runif(1), 3) 



gdOutput<-gradientDescent(indVariables, revenue, initialThetas, 0.000000000001, 20000)



print(gdOutput$coefficients)


plot(gdOutput$costs, xlab="number of repetitions", ylab="J(??)" )


linear.regression.model<-lm(FoodExpenditure ~ Income + FamilySize, data=HouseholdData)
print( linear.regression.model$coefficients )
summary(linear.regression.model)

