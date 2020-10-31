##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)



#PREPROCESSING OF DATA						

#create a column to convert timestamp to date for both edx and validation
edx <- mutate(edx, date = as_datetime(timestamp)) %>% mutate(date = round_date(date, unit = "week"))
validation <- mutate(validation, date = as_datetime(timestamp)) %>% mutate(date = round_date(date, unit = "week"))

#DATA ANALYSIS					
dim(edx) #dimension of the data						
glimpse(edx) #peek of the data						
sapply(edx, class) #class of the attributes						
sapply(edx, n_distinct) #numbers of element in each variables						
#breakdown of the instances in each class						
percentage <- prop.table(table(edx$rating))*100						
cbind(freq = table(edx$rating),  percentage = percentage)						
barchart(percentage) #shows the distribution for each level  of rating						
summary(edx) #shows statistical details of all variables						
memory.limit(56000) #to increase the memory limit						


#EVALUATE MODEL					
# create train and test set. Test set will be 20% of edx data						
set.seed(1) 						
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)						
train <- edx[-test_index,]						
test_set <- edx[test_index,]						

# Make sure userId and movieId in test set are also in train set						
test <- test_set %>% 						
  semi_join(train, by = "movieId") %>%						
  semi_join(train, by = "userId")	

# Add rows removed from test set back into train set						
removed <- anti_join(test_set, test)						
train <- rbind(train, removed)



#Using the Recommendation System and RMSE as metric to select best model 						

#simple average - baseline						
mu <- mean(train$rating)						

naive_rmse <- RMSE(test$rating, mu)						


#adding the movie effect - model_1						
mu <- mean(train$rating)						

movie_avgs <- train %>% 						
  group_by(movieId) %>% 						
  summarize(b_i = mean(rating - mu))						

#the prediction of rating from the test set						
predicted_ratings <- test %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  mutate(pred = mu + b_i) %>%						
  pull(pred)						

#The RMSE of predicted rating to the test rating						
model_1_rmse <- RMSE(predicted_ratings, test$rating)						


#adding the user effect - model_2						
mu <- mean(train$rating)						

user_avgs <- train %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  group_by(userId) %>%						
  summarize(b_u = mean(rating - mu - b_i))						

#the prediction of rating from the test set						
predicted_ratings <- test %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by='userId') %>%						
  mutate(pred = mu + b_i + b_u) %>%						
  pull(pred)						

#The RMSE of predicted rating to the test rating						
model_2_rmse <- RMSE(predicted_ratings, test$rating)						


#adding the genres(b_g) effect - model_3						
mu <- mean(train$rating)						

genres_avgs <- train %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by= 'userId') %>%						
  group_by(genres) %>%						
  summarize(b_g = mean(rating - mu - b_i - b_u))						

#the prediction of rating from the test set						
predicted_ratings <- test %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by='userId') %>%						
  left_join(genres_avgs, by= 'genres') %>%						
  mutate(pred = mu + b_i + b_u + b_g) %>%						
  pull(pred)						

#The RMSE of predicted rating to the test rating						
model_3_rmse <- RMSE(predicted_ratings, test$rating)						


#adding the date(b_t) effect - model_4						
mu <- mean(train$rating)						

date_avgs <- train %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by= 'userId') %>%						
  left_join(genres_avgs, by= 'genres') %>%						
  group_by(date) %>%						
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))						

#the prediction of rating from the test set						
predicted_ratings <- test %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by='userId') %>%						
  left_join(genres_avgs, by= 'genres') %>%						
  left_join(date_avgs, by= 'date') %>%						
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%						
  pull(pred)						

#The RMSE of predicted rating to the test rating						
model_4_rmse <- RMSE(predicted_ratings, test$rating)						


#Selecting the best model with lowest rmse						

model <- c("baseline", "model_1", "model_2", "model_3", "model_4")						
rmses <- c(naive_rmse, model_1_rmse, model_2_rmse, model_3_rmse, model_4_rmse)
data.frame(model, rmses)
qplot(model,rmses)
model[which.min(rmses)]


#The FINAL MODEL						
#The final model will use edx to make predictions from validation.						

#Model_4						
mu <- mean(edx$rating)						

movie_avgs <- edx %>% 						
  group_by(movieId) %>% 						
  summarize(b_i = mean(rating - mu))						

user_avgs <- edx %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  group_by(userId) %>%						
  summarize(b_u = mean(rating - mu - b_i))						

genres_avgs <- edx %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by= 'userId') %>%						
  group_by(genres) %>%						
  summarize(b_g = mean(rating - mu - b_i - b_u))						

date_avgs <- edx %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by= 'userId') %>%						
  left_join(genres_avgs, by= 'genres') %>%						
  group_by(date) %>%						
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))						


#the prediction of rating from the validation						
final_predicted_ratings <- validation %>% 						
  left_join(movie_avgs, by='movieId') %>%						
  left_join(user_avgs, by='userId') %>%						
  left_join(genres_avgs, by= 'genres') %>%						
  left_join(date_avgs, by= 'date') %>%						
  mutate(pred = mu + b_i + b_u + b_g + b_t) %>%						
  pull(pred)						


#The RMSE of predicted ratings to the validation rating						
RMSE <- RMSE(final_predicted_ratings, validation$rating)						

#The FINAL RMSE						
RMSE						



