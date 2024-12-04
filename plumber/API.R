
# Loading required libraries
library(plumber)
library(tidymodels)
library(forcats)
library(dplyr)
library(readr)
library(tidyverse)
library(ggplot2)
library(yardstick)

# Reading in Data
data_api <- read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

# Selecting desired vairables
data_api <- data_api |>
  select(Diabetes_binary, HighBP, HighChol, CholCheck, BMI, PhysActivity, Sex, Age)

# Applying transformations to dataset
data_api <- data_api |>
  mutate(
    # Converting most variables to factors
    diabetes = as.factor(Diabetes_binary),
    high_bp = as.factor(HighBP),
    high_chol = as.factor(HighChol),
    chol_check = as.factor(CholCheck),
    physical_act = as.factor(PhysActivity),
    sex = as.factor(Sex),
    age = as.factor(Age),
    
    # Recoding factor levels to change the names of the values 
    diabetes = fct_recode(diabetes,
                          `No Diabetes` = "0",
                          Diabetes = "1"),
    high_bp = fct_recode(high_bp,
                         No = "0",
                         Yes = "1"),
    high_chol = fct_recode(high_chol,
                           No = "0",
                           Yes = "1"),
    chol_check = fct_recode(chol_check,
                            No = "0",
                            Yes = "1"),
    physical_act = fct_recode(physical_act,
                              No = "0",
                              Yes = "1"),
    sex = fct_recode(sex,
                     Female = "0",
                     Male = "1"),
    age = fct_recode(age,
                     `18-24` = "1",
                     `25-29` = "2",
                     `30-34` = "3",
                     `35-39` = "4",
                     `40-44` = "5",
                     `45-49` = "6",
                     `50-54` = "7",
                     `55-59` = "8",
                     `60-64` = "9",
                     `65-69` = "10",
                     `70-74` = "11",
                     `75-79` = "12",
                     `80_or_older` = "13")) |>
  # Removing most original variables
  select(-Diabetes_binary, -HighBP, -HighChol, -CholCheck, -PhysActivity, -Sex, -Age)

# Setting a seed to make things reproducible
set.seed(23)

# tidy models to split the data into a training and test set (70/30 split)
data_split_api <- initial_split(data_api, prop = 0.70)
data_train_api <- training(data_split)
data_test_api <- testing(data_split)

# 5 fold cross validation on the training set
data_5_fold_api <- vfold_cv(data_train, 5)

# Creating a recipe for data processing
rec_api <- recipe(diabetes ~ ., data = data_api) |>
  step_dummy(high_bp, high_chol, chol_check, physical_act, sex, age) |>
  step_normalize(BMI)

# Random forest model instance
rf_spec_api <- rand_forest(mtry = tune()) |> # varying values for mrty
  set_engine("ranger", importance = "impurity") |>
  set_mode("classification") # To classify best predictor

# Random forest workflow
rf_wkf_api <- workflow() |>
  add_recipe(rec_api) |> # Recipe
  add_model(rf_spec_api) # Defined model instance

# Random forest model fit with tuning parameters and CV
rf_fit_api <- rf_wkf_api |>
  tune_grid(resamples = data_5_fold_api,
            grid = 10,
            metrics = metric_set(mn_log_loss)) # defining model metrics

# Collecting model metrics
rf_fit_api |>
  collect_metrics() |> # Collecting defined model metrics
  filter(.metric == "mn_log_loss") |>
  arrange(mean) # Arranged by mean log-loss value

# Selecting the best Random Forest model 
rf_best_params_api <- rf_fit_api |>
  select_best(metric = "mn_log_loss") # defining the metric

# Finalizing our model on the training set by fitting this chosen model via finalize_workflow()
rf_final_wkf_api <- rf_wkf_api |>
  finalize_workflow(rf_best_params_api)

# Fitting best Random Forest model to entire data set
rf_final_fit_api <- rf_final_wkf_api |>
  last_fit(data_api, metrics = metric_set(mn_log_loss)) # defining model metrics

#* @apiTitle Random Forest Prediction API
#* @apiDescription A plumber build API that displays my best fitting Random Forest model on a subset of the CDC's Behavior Risk Factor Surveillance System data from 2015.

#* Predict diabetes risk
#* @param high_bp The high blood pressure status (default: "Yes")
#* @param high_chol The high cholesterol status (default: "Yes")
#* @param chol_check Cholesterol check status (default: "Yes")
#* @param physical_act Physical activity status (default: "Yes")
#* @param sex Sex of the individual (default: "Female")
#* @param age Age group of the individual (default: "18-24")
#* @param BMI Body Mass Index (default: 30)
#* @post /predict
function(high_bp = "Yes", high_chol = "Yes", chol_check = "Yes",
         physical_act = "Yes", sex = "Female", age = "18-24", BMI = 30) {
  
  # Creating a data frame with the input values
  input_data <- tibble(
    high_bp = as.factor(high_bp),
    high_chol = as.factor(high_chol),
    chol_check = as.factor(chol_check),
    physical_act = as.factor(physical_act),
    sex = as.factor(sex),
    age = as.factor(age),
    BMI = as.numeric(BMI)
  )
  
  # Using the trained model to predict
  prediction <- predict(rf_final_fit_api$.workflow[[1]], input_data, type = "prob")
  
  # Returning the prediction
  list(probabilities = prediction)
}

# Example function calls to test the endpoint:
# 1. curl -X POST "http://127.0.0.1:8000/predict" -d "BMI=28.5&high_bp=No&high_chol=Yes&chol_check=Yes&physical_act=No&sex=Male&age=45-49"
# 2. curl -X POST "http://127.0.0.1:8000/predict" -d "BMI=31.0&high_bp=Yes&high_chol=Yes&chol_check=No&physical_act=Yes&sex=Female&age=35-39"
# 3. curl -X POST "http://127.0.0.1:8000/predict" -d "BMI=27.0&high_bp=No&high_chol=No&chol_check=Yes&physical_act=Yes&sex=Male&age=18-24"




#* API Info
#* @get /info
function() {
  list(
    message = "Hello, my name is Ethan Marburger\ 
    The folowing URL provides more information about the dataset as well as determining the best fitting model [TO DO: RENDER SITES AND CREATE URL]."
  )
}




#* Confusion Matrix Plot
#* @get /confusion
#* @serializer png
function() {
  # Generate predictions for the entire dataset
  predictions <- predict(rf_final_fit_api$.workflow[[1]], data_api) |> 
    bind_cols(data_api) |> 
    mutate(.pred_class = as.factor(.pred_class))
  
  # Create the confusion matrix
  conf_mat <- conf_mat(predictions, truth = diabetes, estimate = .pred_class)
  
  # Create a confusion matrix plot
  conf_mat_plot <- autoplot(conf_mat, type = "heatmap") +
    labs(title = "Confusion Matrix for Best Model",
         subtitle = "Comparing Predictions to Actual Values") +
    theme_minimal()
  
  # Return the plot as a PNG image
  print(conf_mat_plot)
}
