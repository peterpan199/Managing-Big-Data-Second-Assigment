#Topic 6


library(quantmod)
library(AER)
library(ggplot2)


#setting working directory and importing data set

setwd("C:/metaptyxiako/diaxeirhsh megalwn dedomenwn/Assigment2/ergasia/ergasia")
communitiesdata <- read.csv("C:/metaptyxiako/diaxeirhsh megalwn dedomenwn/Assigment2/ergasia/ergasia/communities.data", header=FALSE,stringsAsFactors = TRUE)
class(communitiesdata)
#Number Of Observations
numObs<- nrow(communitiesdata)



randomobservations <- sample(communitiesdata)
# Πλήθος παρατηρήσεων
 ViolentCrimesPerPop <- function(x,y ,theta){
  
     return (sum((x%*%theta- y)^2)/(2*numObs))
  
  }

 gradientDescent<-function(x, y, theta, alpha=0.01, numIters=90){
   
 
  
   CrimesHistory <- rep(0, numIters)
    
    for(k in 1:numIters){
     
    
      for (i in 1:numObs){
        for (j in 1:10){
          
          
          
        
          newtheta<-theta[j] - alpha * (x[i, j]*(x[i, j] * theta[j] - y[i]))
          theta[j]<-newtheta
          
        }
        
        CrimesHistory[i]  <- ViolentCrimesPerPop(x, y, theta)
        
         
     }
   
    }
  
   gdResults<-list("coefficients"=theta, "ViolentCrimes"= CrimesHistory)
   return(gdResults)
 }
 
 y <- communitiesdata$V128 
 x <- cbind(rep(1, numObs),communitiesdata$V18,communitiesdata$V27,communitiesdata$V28,communitiesdata$V32,communitiesdata$V33,communitiesdata$V38,communitiesdata$V91,communitiesdata$V77,communitiesdata$V96)
 transX<- t(x)                           
 theta <- matrix(runif(n= 10 , min= 1 , max= 1 ), nrow= 10 )
 

 
 
 numIters = 100
 alpha  = 0.01

 gdOutput <- gradientDescent(x,y, theta, alpha, numIters)
 gdOutput
 
 plot(gdOutput$ViolentCrimes, ylab="J(θ)", xlab="Iterations")

 # Εμφάνιση συντελεστών που εκτιμήθηκαν
 print(gdOutput$coefficients)
 warnings()

 
