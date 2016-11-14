Background
----------

Six young health participants were asked to perform one set of 10
repetitions of the Unilateral Dumbbell Biceps Curl in five different
fashions: exactly according to the specification (Class A), throwing the
elbows to the front (Class B), lifting the dumbbell only halfway (Class
C), lowering the dumbbell only halfway (Class D) and throwing the hips
to the front (Class E).

Class A corresponds to the specified execution of the exercise, while
the other 4 classes correspond to common mistakes. Participants were
supervised by an experienced weight lifter to make sure the execution
complied to the manner they were supposed to simulate. The exercises
were performed by six male participants aged between 20-28 years, with
little weight lifting experience. We made sure that all participants
could easily simulate the mistakes in a safe and controlled manner by
using a relatively light dumbbell (1.25kg). The goal of your project is
to predict the manner in which they did the exercise. Read more:
<http://groupware.les.inf.puc-rio.br/har#ixzz4P4AW29ar>

Data Preparation
----------------

Firstly, we load the training and testing data from the physical
location into r enviroment.

    library(caret)
    library(randomForest)

    trainingset <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!", ""));
    testingset <-read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!", ""));

### Clean up data

Before we use the model, we need to clean the data first. Firstly we
omit the columns that contain na/null data, then we remove the timestamp
columns and all other character type columns. This procedure was done to
both training and testing set.

    #training set cleanup
    training <-trainingset[,colSums(is.na(trainingset)) == 0]
    col.name <- names(training)
    col.name <- col.name[-grep("*timestamp",col.name)]    
    training<-training[,col.name];
    training<-training[,-c(1:3)];

    #testing set cleanup
    testing <-testingset[,colSums(is.na(testingset)) == 0]
    col.name <- names(testing)
    col.name <- col.name[-grep("*timestamp",col.name)]    
    testing<-testing[,col.name];
    testing<-testing[,-c(1:3)];
    names(testing)

    ##  [1] "num_window"           "roll_belt"            "pitch_belt"          
    ##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
    ##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
    ## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
    ## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
    ## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
    ## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
    ## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
    ## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
    ## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
    ## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
    ## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
    ## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
    ## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
    ## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
    ## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
    ## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
    ## [52] "magnet_forearm_y"     "magnet_forearm_z"     "problem_id"

### Partitioning the data

We split the training data into 75% for training and 25% for testing, as
the provided testing data will be use for validation.

    inTrain <- createDataPartition(training$classe,p=0.75,list=FALSE);
    train.data <- training[inTrain,];
    test.data <- training[-inTrain,];

Model Fitting
-------------

We will use random forest method with cross validation with 15 fold as
the train control to increase the accuracy.

    set.seed(11115);
    ctrl<-trainControl(method="cv",number=15);
    fit.ctrl <- train(data=train.data,train.data$classe~.,method="rf",trControl=ctrl);
    fit.ctrl

    ## Random Forest 
    ## 
    ## 14718 samples
    ##    53 predictors
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (15 fold) 
    ## 
    ## Summary of sample sizes: 13737, 13737, 13736, 13736, 13737, 13738, ... 
    ## 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
    ##   2     0.995     0.994  0.0017       0.00215 
    ##   27    0.998     0.997  0.00155      0.00196 
    ##   53    0.995     0.993  0.00214      0.00271 
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

From the result we can see the accuracy is in 99.8%, next we will see
the accuracy from the test set that we have defined before.

Prediction and result
---------------------

    pred.fit.ctrl <- predict(fit.ctrl,test.data);
    confusionMatrix(test.data$classe,pred.fit.ctrl)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1395    0    0    0    0
    ##          B    3  946    0    0    0
    ##          C    0    0  855    0    0
    ##          D    0    0    2  802    0
    ##          E    0    0    0    3  898
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9984          
    ##                  95% CI : (0.9968, 0.9993)
    ##     No Information Rate : 0.2851          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9979          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9979   1.0000   0.9977   0.9963   1.0000
    ## Specificity            1.0000   0.9992   1.0000   0.9995   0.9993
    ## Pos Pred Value         1.0000   0.9968   1.0000   0.9975   0.9967
    ## Neg Pred Value         0.9991   1.0000   0.9995   0.9993   1.0000
    ## Prevalence             0.2851   0.1929   0.1748   0.1642   0.1831
    ## Detection Rate         0.2845   0.1929   0.1743   0.1635   0.1831
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
    ## Balanced Accuracy      0.9989   0.9996   0.9988   0.9979   0.9996

From the result, we can see the out of sample error which is only around
0.21% chance of error, so the accuracy is quite good. Below script will
generate the files for the testing prediction result.

    result <- predict(fit.ctrl,testing)
    write_files = function(x){
      n = length(x)
      for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
      }
    }

    write_files(result)
