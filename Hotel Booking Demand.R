library(pROC)
library(dplyr)
library(nnet)
library(e1071)
library(NeuralNetTools)

## open the data file
hotel_booking= read.csv("hotel_bookings.csv")

## Since it is too large to run on local computer, we select 2000 data to use ramdonly.
booking= sample_n(hotel_booking, 10000,replace = FALSE)

## Check the summary of the dataset
summary(booking)

## is_canceled is a categorical variable which is created based on the variable reservation_status
booking$is_canceled= as.factor(booking$is_canceled)
booking$reservation_status=NULL

## The variable company has more than 90% NULL values (9391 NULL values), so it will not be used
booking$company=NULL

## check the dataset again
summary(booking)

## Create the variable reserved_vs_assigned. If the reserved room type is as same as the assigned room type, 
## reserved_vs_assigned will be 1, otherwise it will be 0.
booking$reserved_vs_assigned=rep(NA,10000)
for (i in 1:10000) {
  booking$reserved_vs_assigned[i]=ifelse(booking$reserved_room_type[i]==booking$assigned_room_type[i],1,0)
}

## check the propotion of the two section in the variable reserved_vs_assigned
booking$reserved_vs_assigned=as.factor(booking$reserved_vs_assigned)
table(booking$reserved_vs_assigned)

## remove the variables reserved_room_type and assigned_room_type
booking$reserved_room_type=NULL
booking$assigned_room_type=NULL


## combine year, month, day as a date
booking$arrival_date <- as.Date(with(booking, 
                                     paste(arrival_date_year, arrival_date_month, arrival_date_day_of_month,sep="-")), "%Y-%B-%d")
## remove year,month, day
booking[,4:7]=NULL

## convert reservation date to a date variable
booking$reservation_status_date=as.Date(booking$reservation_status_date)

## check the relationship between the resevation date and cancellation
plot(booking$reservation_status_date,booking$is_canceled)
## check the relationship between the arrival date and cancellation
plot(booking$arrival_date,booking$is_canceled)

## there is no significant correlation between the two kinds of dates and bookings cancellation, 
##so the two date variables are removed 
booking$arrival_date=NULL
booking$reservation_status_date=NULL

## check the dataset again
summary(booking)

## Since angent has 334 levels and country has 178 levels, too many varieties to use, they are removed.
length(table(booking$agent))
length(table(booking$country))
booking$agent=NULL
booking$country=NULL

### check the bookings cancellation rate of the dataset
sum(booking$is_canceled==1)/10000


###################################################################
##### Double cross-validation for modeling-process assessment #####				 
###################################################################

##### model assessment OUTER shell #####
fulldata.out = booking
k.out = 10 
n.out = dim(fulldata.out)[1]
#define the cross-validation splits 
groups.out = c(rep(1:k.out,floor(n.out/k.out))); if(floor(n.out/k.out) != (n.out/k.out)) groups.out = c(groups.out, 1:(n.out%%k.out))
set.seed(123)
cvgroups.out = sample(groups.out,n.out)  #orders randomly, with seed (123) 
# set up storage for predicted values from the double-cross-validation
allpredictedCV.out = rep(NA,n.out)

# loop through outer splits
for (j in 1:k.out)  { 
  groupj.out = (cvgroups.out == j)
  # define the training set for outer loop
  traindata.out = booking[!groupj.out,]
  #define the validation set for outer loop
  validdata.out = booking[groupj.out,]
  
  ### entire model-fitting process ###
  fulldata.in = traindata.out
  
  ###########################
  ## Full modeling process ##
  ###########################
  
  n.in = dim(fulldata.in)[1]
 # x.in = model.matrix(is_canceled~.,data=fulldata.in)[,-1]
#  y.in = fulldata.in[,2]
  k.in = 10 
  #produce list of group labels
  groups.in = c(rep(1:k.in,floor(n.in/k.in))); if(floor(n.in/k.in) != (n.in/k.in)) groups.in = c(groups.in, 1:(n.in%%k.in))
  cvgroups.in = sample(groups.in,n.in)  
  
  
  #################################################
  ##### cross-validation for model selection #####
  ##################################################
  
  ############## SVM Model ################
  
  ### Perform cross-validation to compare different values of cost and gamma for a radial model.
  
  tune.out.gama = tune(svm, is_canceled~., data = fulldata.in, kernel="linear", type="C-classification", 
                       ranges = list( cost = c( .1, 1, 10,100 ),
                                      gamma = c(0.5, 1, 3, 5)))
  
  
  best.mod.gama = tune.out.gama$best.model
  minCV.svm = sum(fulldata.in$is_canceled != best.mod.gama$fitted)/n.in
  best.gama= best.mod.gama$gamma
  best.cost= best.mod.gama$cost
  
  
  ######### ANN Model ########
  
  ### select the best decay rate from 0.1 to 3.1
  decayRate = seq(0.1,3.1,by=.3)
 
  CV.ann = rep(NA, length(decayRate))
  
  ### Separate numeric variables and categorical variables
  n_variables = subset(fulldata.in, select=-c(hotel, is_canceled, meal, market_segment, distribution_channel,
                                              customer_type,reserved_vs_assigned, deposit_type))
                                             
  
  c_variables=data.frame(hotel=fulldata.in$hotel, is_canceled= fulldata.in$is_canceled, meal=fulldata.in$meal, market_segment= fulldata.in$market_segment,
                         distribution_channel=fulldata.in$distribution_channel, customer_type= fulldata.in$customer_type,
                         reserved_vs_assigned=fulldata.in$reserved_vs_assigned, deposit_type=fulldata.in$deposit_type) 
                         
  
  
  #### standardize all of the numeric variables to have mean 0 and standard deviation 1.
  #### to fit an artificial neural network with 10 hidden nodes
  ### Use 10-fold cross-validation to choose the best decay rate
  
  for(i in 1:k.in){
    groupi = (cvgroups.in == i)
    H.train = scale(n_variables[!groupi, ])
    H.valid = scale(n_variables[groupi, ], center = attr(H.train, "scaled:center"), 
                    scale = attr(H.train, "scaled:scale"))
    tr= cbind(H.train, c_variables[!groupi, ])
    va =cbind(H.valid, c_variables[groupi, ])
    CV.ann = rep(NA, length(decayRate))
    
    for(j in 1:length(decayRate)){
      fit = nnet(is_canceled ~ ., data=tr, size=10,decay= decayRate[j], maxit=1000,
                 trace = F)
      pre= ifelse(predict(fit, va)>0.5,2,1)
      CV.ann[j]  = sum(as.numeric(va$is_canceled) != pre)/n.in
    } 
  }
  
  ### compute the lowest CV
  minCV.ann = min(CV.ann)
  
  ### compute which decayrate has the lowest CV
  CV.ann[j]=min(CV.ann)
  best.decayrate= decayRate[j]
  
  
  #################################
  #### End of modeling process ####
  #################################
  
  ### finally, fit the best model to the full (available) data ###
  
  ## if the cv of svm model is smaller than that of that of ann model
  if (minCV.svm<= minCV.ann){
    ## use the svm model with best gamma to predict validation data 
    newX = subset(validdata.out,select=-c(is_canceled))
    predictedCV=predict(best.mod.gama, newX)
    
  ## if the cv of ann model is smaller than that of that of svm model
  }else if(minCV.svm >minCV.ann){
    
    ### Separate numeric variables and categorical variables
    n_variables.out= subset(validdata.out, select=-c(hotel, is_canceled, meal, market_segment, distribution_channel,
                                                     customer_type,reserved_vs_assigned, deposit_type))
                                                  
    
    c_variables.out=data.frame(hotel=validdata.out$hotel, is_canceled= validdata.out$is_canceled, meal=fulldata.in$meal,market_segment= validdata.out$market_segment,
                               distribution_channel=validdata.out$distribution_channel, customer_type= validdata.out$customer_type,
                               reserved_vs_assigned=validdata.out$reserved_vs_assigned, deposit_type=validdata.out$deposit_type )
    
    
    ###  standardize all of the numeric variables
    t.out= scale(n_variables)
    v.out= scale(n_variables.out,center = attr(t.out, "scaled:center"), 
                 scale = attr(t.out, "scaled:scale"))
    
    tr.out= cbind(t.out, c_variables)
    va.out =cbind(v.out, c_variables.out)
    
    #### use the ann model with best decayrate to predict validation data
    fit.out= nnet(is_canceled~., data=tr.out, size=10,decay= best.decayrate, maxit=1000,
                  trace = F)
    predicted =predict(fit.out, va.out)
    predictedCV = ifelse(predicted>0.5, 2,1)
    
  }
  predictedCV =  as.numeric(predictedCV) -1 
  allpredictedCV.out [groupj.out] = predictedCV
}  

##assessment
## create confusion matrix
table(booking$is_canceled,allpredictedCV.out)
CV10.out = sum(booking$is_canceled!=allpredictedCV.out)/n.out
##the proportion of correct classifications 
p.out = 1-CV10.out; p.out 

### create ROC curve
myroc = roc(booking$is_canceled,allpredictedCV.out)
plot.roc(myroc)
## check the area under curve
auc(myroc)

####################################
#### Check The Variables Importance
#####################################

## use garson algorithm to compute variable importance
garson(fit.out)
g=garson(fit.out)
## check the weight of variable importance
g$data


## use olden algorithm to compute variable importance 
olden(fit.out)
o=olden(fit.out)
## check the weight of variable importance 
o$data