import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')



#
# Multiply two matrices i.e. mat1 * mat2
#
def matmultiply(mat1,mat2):
    
    return( np.matmul(mat1, mat2) )
    


#
# Calculate current value of cost function J(θ).
# indV: matrix of independent variables, first column must be all 1s
# depV: matrix (dimensions nx1)of dependent variable i.e.
#
def calculateCost(indV, depV, thetas):
    return( np.sum( ((matmultiply(indV, thetas) - depV)**2) / (2*indV.shape[0]) ) )  
    

#
# Batch gradient descent
#
# indV:matrix of independent variables, first column must be all 1s
# depV: matrix (dimensions nx1)of dependent variable i.e.
# alpha: value of learning hyperparameter. Default (i.e. if no argument provided)  0.01
# numIters: number of iterations. Default (i.e. if no argument provided) 100
#
def batchGradientDescent(indV, depV, thetas, alpha=0.1, numIters=200, verbose=False):
    calcThetas = thetas
    costHistory = pd.DataFrame(columns=["iter", "cost"])
    m = len(depV)
    for i in range(0, numIters):
        prediction = np.dot(indV, calcThetas)
        calcThetas = calcThetas - (1 / m) * alpha * (indV.T.dot(prediction - depV))
        print(">>>> Iteration", i, ")")
        print("       Calculate thetas...", calcThetas)
        c = calculateCost(indV, depV, calcThetas)
        print("       Calculate cost fuction for new thetas...", c)
        costHistory = costHistory.append({"iter": i, "cost": c}, ignore_index=True)
    return calcThetas, costHistory



communities = pd.read_csv("communities.data", header=None, sep=",", engine='python')
communities = communities.set_axis(["state", "county", "community", "communityname", "fold", "population", "householdsize", "racepctblack", "racePctWhite", "racePctAsian",
                           "racePctHisp", "agePct12t21", "agePct12t29", "agePct16t24", "agePct65up", "numbUrban", "pctUrban", "medIncome", "pctWage", "pctWFarmSelf", 
                           "pctWInvInc", "pctWSocSec", "pctWPubAsst", "pctWRetire", "medFamInc", "perCapInc",
                           "whitePerCap", "blackPerCap", "indianPerCap", "AsianPerCap", "OtherPerCap", "HispPerCap", "NumUnderPov", "PctPopUnderPov", "PctLess9thGrade",
                           "PctNotHSGrad", "PctBSorMore", "PctUnemployed", "PctEmploy", "PctEmplManu", "PctEmplProfServ", "PctOccupManu", "PctOccupMgmtProf", "MalePctDivorce",
                           "MalePctNevMarr", "FemalePctDiv", "TotalPctDiv", "PersPerFam", "PctFam2Par", "PctKids2Par", "PctYoungKids2Par", "PctTeen2Par", "PctWorkMomYoungKids",
                           "PctWorkMom", "NumIlleg", "PctIlleg", "NumImmig", "PctImmigRecent", "PctImmigRec5", "PctImmigRec8", "PctImmigRec10", "PctRecentImmig", "PctRecImmig5",
                           "PctRecImmig8", "PctRecImmig10", "PctSpeakEnglOnly", "PctNotSpeakEnglWell", "PctLargHouseFam", "PctLargHouseOccup", "PersPerOccupHous", "PersPerOwnOccHous",
                           "PersPerRentOccHous", "PctPersOwnOccup", "PctPersDenseHous", "PctHousLess3BR", "MedNumBR", "HousVacant", "PctHousOccup", "PctHousOwnOcc", "PctVacantBoarded", 
                           "PctVacMore6Mos", "MedYrHousBuilt", "PctHousNoPhone", "PctWOFullPlumb", "OwnOccLowQuart", "OwnOccMedVal", "OwnOccHiQuart", "RentLowQ", "RentMedian",
                           "RentHighQ", "MedRent", "MedRentPctHousInc", "MedOwnCostPctInc", "MedOwnCostPctIncNoMtg", "NumInShelters", "NumStreet", "PctForeignBorn", "PctBornSameState",
                           "PctSameHouse85", "PctSameCity85", "PctSameState85", "LemasSwornFT", "LemasSwFTPerPop", "LemasSwFTFieldOps", "LemasSwFTFieldPerPop", "LemasTotalReq", 
                           "LemasTotReqPerPop", "PolicReqPerOffic", "PolicPerPop", "RacialMatchCommPol", "PctPolicWhite", "PctPolicBlack", "PctPolicHisp", "PctPolicAsian", 
                           "PctPolicMinor", "OfficAssgnDrugUnits", "NumKindsDrugsSeiz", "PolicAveOTWorked", "LandArea", "PopDens", "PctUsePubTrans", "PolicCars", "PolicOperBudg", 
                           "LemasPctPolicOnPatr", "LemasGangUnitDeploy", "LemasPctOfficDrugUn", "PolicBudgPerPop", "ViolentCrimesPerPop"], axis=1)

dependentVar = communities.iloc[:, 127]

# These are all our independent ones: 17,26,27,31,32,37,76,90,95
# NOTE: above indecies are 0-based!
independentVars = communities.iloc[:, [17,26,27,31,32,37,76,90,95] ]

# Check to see if missing values are present.
# NOTE: ? signify missing values
independentVars = independentVars[(independentVars != '?').all(1)]

# Add new column at the beginning representing the constant term b0
independentVars.insert(0, 'b0', 1)




# Initialize thetas with some random values.
# We'll need (independentVars.shape[1])  theta values, one for each independent variable.
# NOTE: First theta value is coefficient for variable in FIRST column in independent matrix independentVars, second theta variable is coefficient
#       for second column in independent matrix independentVars etc
iniThetas = []
for i in range(0, independentVars.shape[1]):
    iniThetas.append( np.random.rand() )

initialThetas = np.array(iniThetas)

# Everything is ok.
# Run BATCH gradient descent and return 2 values: I) the vector of the estimated coefficients (estimatedCoefficients) and II) the values of the
# cost function (costHistory)
estimatedCoefficients, costHistory = batchGradientDescent(independentVars.to_numpy(), dependentVar.to_numpy(), initialThetas, 0.1, 200)

print(estimatedCoefficients)

# Display now the cost function to see if alpha and number of iterations were appropriate.
costHistory.plot.scatter(x="iter", y="cost", color='red')
plt.show()


