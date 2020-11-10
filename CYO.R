
##########################################################
    # CHOOCE YOUR OWN PROJECT - FERTILITY DIAGNOSIS #
##########################################################


# 1. Install all necessary packages

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(readr)


# 2. Download and load the data set

# 2.1 The URL to download the data
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt"
# 2.2 Load the data and name as fertility
fertility <- read_csv(url, col_names = FALSE)
# 2.3 set the column names
colnames(fertility) <- c("Season", "Age", "Diseases", "Accident", "Surgical", "Fevers", "Alcohol", "Smoking", "Sitting", "Diagnosis")
                                

## 3. Attributes Information and names:

#Season - Season in which the analysis was performed. 1) winter, 2) spring, 3) Summer, 4) fall. (-1, -0.33, 0.33, 1)
#Age - Age at the time of analysis. 18-36 (0, 1)
#Diseases - Childish diseases (ie , chicken pox, measles, mumps, polio) 1) yes, 2) no. (0, 1)
#Accident - Accident or serious trauma 1) yes, 2) no. (0, 1)
#Surgical - Surgical intervention 1) yes, 2) no. (0, 1)
#Fevers - High fevers in the last year 1) less than three months ago, 2) more than three months ago, 3) no. (-1, 0, 1)
#Alcohol - Frequency of alcohol consumption 1) several times a day, 2) every day, 3) several times a week, 4) once a week, 5) hardly ever or never (0, 1)
#Smoking - Smoking habit 1) never, 2) occasional 3) daily. (-1, 0, 1)
#Sitting - Number of hours spent sitting per day - 16 (0, 1)
#Diagnosis - Output: Diagnosis normal (N), altered (O)


# 4. Summarize Data set

# 4.1 Dimensions of Data set
dim(fertility)
# 4.2 Types of attributes
sapply(fertility, class)
# 4.3 Glimpse at the data
glimpse(fertility)
# 4.4 Number of instances in each attribute
sapply(fertility, n_distinct)
# 4.5 Statistical summary of the attributes
summary(fertility)


# 5. Visualization and graphic explanation of data set

# 5.1 split predictors and output
x <- fertility[,1:9]
y <- fertility[,10]
# 5.2 barplot to show the distribution of the class
barchart(y)
# 5.3 boxplot for each attribute on one image (1 - 5)
par(mfrow=c(1,5))
   for(i in 1:5) {
     boxplot(x[,i], main = names(fertility)[i])
   }

    for(i in 6:9) {
      boxplot(x[,i], main = names(fertility)[i])
   }


# 6. create the train and validation sets 

# Validation set will be 20% of Fertility data
set.seed(1, sample.kind="Rounding") # set the seed
test_index <- createDataPartition(y = fertility$Diagnosis, times = 1, p = 0.2, list = FALSE)
train <- fertility[-test_index,]
validation <- fertility[test_index,]


# 7. Evaluate Algorithms

# An insight - A guess to the accuracy by simple sampling
y_hat <- sample(c("N", "O"), length(test_index), replace = TRUE) # simple sampling for y
mean(y_hat == validation$Diagnosis) # Accuracy of the sampling

# The following 3 Algorithms will be used and overall accuracy will be used as metric

# 7.1 Linear Discriminant Analysis (LDA) - Linear
set.seed(1)
train.lda <- train(Diagnosis ~ ., data = train, method = "lda") # train with lda
predict.lda <- predict(train.lda, validation) # classification from validation set
Accuracy.lda <- mean(predict.lda == validation$Diagnosis) # overall accuracy of classification
Accuracy.lda

# 7.2 Kernel Nearest Neighbors (kNN) - Non Linear
set.seed(1)
train.knn <- train(Diagnosis ~ ., data = train, method = "knn") # train with knn
predict.knn <- predict(train.knn, validation) # classification from validation set
Accuracy.knn <- mean(predict.knn == validation$Diagnosis) # overall accuracy of classification
Accuracy.knn

# 7.3 Random Forest (rf) - Advanced
set.seed(1)
train.rf <- train(Diagnosis ~ ., data = train, method = "rf") # train with random forest
predict.rf <- predict(train.rf, validation) # classification from validation set
Accuracy.rf <- mean(predict.rf == validation$Diagnosis) # overall accuracy of classification
Accuracy.rf

# 8. Results analysis of the Algorithms

# 8.1 Analysis of each Algorithm
# 8.1.1 Linear Discriminant Analysis (LDA) - Linear
print(train.lda) # train summary
train.lda$finalModel # variable importance to each class

# 8.1.2 Kernel Nearest Neighbors (kNN) - Non Linear
print(train.knn) # analysis of the train
ggplot(train.knn) # graphical representation


# 8.1.3 Random Forest (rf) - Advanced
print(train.rf) # Analysis of the train
ggplot(train.rf) # graphical representation
varImp(train.rf) # The shows variable importance 


# 8.2 Comparison of Algorithms

# 8.2.1 Training comparison of Algorithms
results_train <- resamples(list(lda = train.lda, knn = train.knn, rf = train.rf)) # list of Algorithms used
summary(results_train) # summary of the Algorithms train
dotplot(results_train) # graphical presentation of the train summary
ggplot(results_train) # another graphical presentation

# 8.2.2 Comparison of Algorithms Accuracy
model <- c("lda", "knn", "rf")	# list of Algorithms used					
Accuracy <- c(Accuracy.lda, Accuracy.knn, Accuracy.rf) # Accuracy of each Algorithm
data.frame(model, Accuracy) # Tabulating the results
qplot(model, Accuracy) # Graphical presentation of the Algorithms performances

# 8.2.3 Confusion Matrix analysis
confusionMatrix(predict.lda, factor(validation$Diagnosis)) # lda confusion matrix
confusionMatrix(predict.knn, factor(validation$Diagnosis)) # knn confusion matrix
confusionMatrix(predict.rf, factor(validation$Diagnosis)) # rf confusion matrix

