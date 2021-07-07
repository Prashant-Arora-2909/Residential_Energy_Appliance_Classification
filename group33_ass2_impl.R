# Package names
packages <- c(
  "caret", "doParallel", "data.table", "tsfeatures", "tidyverse", "xgboost", "gam", "MLeval",
  "LiblineaR", "bnclassify", "C50", "plyr", "ggplot2", "ROSE"
)

# Install packages not yet installed
installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Packages loading
invisible(lapply(packages, library, character.only = TRUE))

# ggplot settings
theme_update(plot.title = element_text(hjust = 0.5))

options(warn = -1)
set.seed(888)
registerDoParallel(4)
getDoParWorkers()


# Some constants
CWD <- getwd()
RESULTS_FOLDER <- paste(CWD)


BASE_COLUMNS <- c(
  "id", "load", "hourofday", "dif", "absdif", "max", "var", "entropy", "nonlinear", "hurst",
  "dayofweek"
)

# Create a named list of 5 data sets for each appliance
APPLIANCE <- list("ac", "ev", "oven", "wash", "dryer")


# Create test and train data sets
df_test <- fread("test_data_nolabels.csv")
df_train <- fread("train_data_withlabels.csv")

names(df_train)[1] <- "id"
names(df_test)[1] <- "id"

# Create a validation set
valdn_index <- 244460:294460
df_valdn <- df_train[valdn_index, ]
fwrite(df_valdn[, c("id", "ac", "ev", "oven", "wash", "dryer")], "valdn.csv")

# Create a table to hold the results of the confusion matrix
create_results_table <- function() {
  df_results <- data.table(
    app = NA,
    model = NA,
    Sensitivity = NA,
    Specificity = NA,
    Pos.Pred.Value = NA,
    Neg.Pred.Value = NA,
    Precision = NA,
    Recall = NA,
    F1 = NA,
    Prevalence = NA,
    Detection.Rate = NA,
    Detection.Prevalence = NA,
    Balanced.Accuracy = NA
  )
  df_results
}


# PREPROCESS #

# Convert into factor for classification
df_train$ac <- as.factor(df_train$ac)
df_train$ev <- as.factor(df_train$ev)
df_train$oven <- as.factor(df_train$oven)
df_train$wash <- as.factor(df_train$wash)
df_train$dryer <- as.factor(df_train$dryer)

levels(df_train$ac) <- c("off", "on")
levels(df_train$ev) <- c("off", "on")
levels(df_train$oven) <- c("off", "on")
levels(df_train$wash) <- c("off", "on")
levels(df_train$dryer) <- c("off", "on")

# Create validation out of 50K rows
df_valdn <- df_train[valdn_index, ]

# Create a list of validation data tables for each appliance
app_valdn <- list(
  ac = df_valdn,
  ev = df_valdn,
  oven = df_valdn,
  wash = df_valdn,
  dryer = df_valdn
)

# Create a list of train data tables for each appliance
app_bal <- list(
  ac = df_train[-valdn_index, -c(4, 5, 6, 7)],
  ev = df_train[-valdn_index, -c(3, 5, 6, 7)],
  oven = df_train[-valdn_index, -c(3, 4, 6, 7)],
  wash = df_train[-valdn_index, -c(3, 4, 5, 7)],
  dryer = df_train[-valdn_index, -c(3, 4, 5, 6)]
)

# Re-balance the response classes
ac <- setDT(ovun.sample(ac ~ ., data = app_bal$ac, method = "both", p = 0.5, N = 70000, seed = 1)$data)
ev <- setDT(ovun.sample(ev ~ ., data = app_bal$ev, method = "both", p = 0.5, N = 9000, seed = 1)$data)
oven <- setDT(ovun.sample(oven ~ ., data = app_bal$oven, method = "both", p = 0.5, N = 40000, seed = 1)$data)
wash <- setDT(ovun.sample(wash ~ ., data = app_bal$wash, method = "both", p = 0.5, N = 15000, seed = 1)$data)
dryer <- setDT(ovun.sample(dryer ~ ., data = app_bal$dryer, method = "both", p = 0.5, N = 2000, seed = 1)$data)

# Update the train list
app_train <- list(ac = ac, ev = ev, oven = oven, wash = wash, dryer = dryer)

# UTILITY FUNCTIONS #

# Create the submission csv
prepare_prediction_csv <- function(dt, text) {
  col_names <- names(dt)[12:16]
  col_new <- c("ac", "ev", "oven", "wash", "dryer")
  setnames(dt, col_names, col_new)


  levels(dt$ac) <- c(0, 1)
  levels(dt$ev) <- c(0, 1)
  levels(dt$oven) <- c(0, 1)
  levels(dt$wash) <- c(0, 1)
  levels(dt$dryer) <- c(0, 1)

  fwrite(dt[, c("id", "ac", "ev", "oven", "wash", "dryer")], paste(text, ".csv", sep = ""))
}


# MODEL FORMULA

# List of baseline functions. I have kept the list structure inplace as this was used 
# to evaluate multiple formula
baseline <- list(ac ~ ., ev ~ ., oven ~ ., wash ~ ., dryer ~ .)

list_of_formula <- list(baseline = baseline)

update_names <- function(x) {
  names(x) <- APPLIANCE
  x
}

list_of_formula <- lapply(list_of_formula, update_names)
baseline <- list_of_formula[[1]]


# trainControl configuration
ts_CV <- trainControl(
  method = "repeatedcv",
  number = 5,
  search = "random",
  verboseIter = TRUE,
  savePredictions = T,
  seeds = NULL,
  allowParallel = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Caret's model metric
metric <- "ROC"

# This is kept as a function. During development multiple model
# runs could be called from this script.

run_job <- function(job_name) {

  # Again during development this was one loop that 
  # ran for each appliance.It has been broken to five 
  # loops each with one iteration. 

  # RUN FOR AC
  for (each_app in APPLIANCE[1]) {
    cat(paste("Fitting for ", each_app, "\n", sep = ""))

    train_subset_df <- app_train[[each_app]][, c(..each_app, ..BASE_COLUMNS)] 

    cat(paste("Fitting XGB "))
    fit_xgb <- train(baseline[[each_app]],
      data = train_subset_df, method = "xgbTree", metric = metric,
      trControl = ts_CV, maximize = FALSE, scale_pos_weight = 1 # added scalepos and maximize.
    )

    # In development this would have contained multiple models, hence the list.
    model_list <- list(xgb = fit_xgb)

    # Predict the raw values for test(no labels) and (validation)
    pred_valdn_list <- predict(model_list, df_valdn, type = "raw")
    pred_test_list <- predict(model_list, df_test, type = "raw")

    # Create "appliance_model" column names for use in the loop.
    col_names <- paste(each_app, names(model_list), sep = "_")

    # Create the confusion matrix against the validation data set
    for (each_model in names(model_list)) {
      check_f1 <- confusionMatrix(pred_valdn_list[[each_model]], df_valdn[[each_app]], positive = "on")
      cat(paste(each_model, " -> ", names(check_f1$byClass), ": ", check_f1$byClass, "\n", sep = ""))

      # Save the results of the confusion matrix to a data.table
      run_data <- data.frame(as.list(c(app = each_app, model = each_model, check_f1$byClass)))
      df_results <- rbind(df_results, run_data)
    }

    # Create a new column in the format of "appliance_model" to save the predictions.
    df_test[, c(col_names) := c(pred_test_list)]
    df_valdn[, c(col_names) := c(pred_valdn_list)]
  }

  # RUN FOR EV
  for (each_app in APPLIANCE[2]) {

    cat(paste("Fitting for ", each_app, "\n", sep = ""))
    train_subset_df <- app_train[[each_app]][, c(..each_app, ..BASE_COLUMNS)]

    cat(paste("Fitting GLM "))
    fit_glm <- train(baseline[[each_app]],
      data = train_subset_df, method = "glm", metric = metric,
      trControl = ts_CV
    )

    model_list <- list(glm = fit_glm)

    pred_valdn_list <- predict(model_list, df_valdn, type = "raw")
    pred_test_list <- predict(model_list, df_test, type = "raw")
    col_names <- paste(each_app, names(model_list), sep = "_")

    for (each_model in names(model_list)) {
      check_f1 <- confusionMatrix(pred_valdn_list[[each_model]], df_valdn[[each_app]], positive = "on")
      cat(paste(each_model, " -> ", names(check_f1$byClass), ": ", check_f1$byClass, "\n", sep = ""))

      run_data <- data.frame(as.list(c(app = each_app, model = each_model, check_f1$byClass)))
      df_results <- rbind(df_results, run_data)
    }

    df_test[, c(col_names) := c(pred_test_list)]
    df_valdn[, c(col_names) := c(pred_valdn_list)]
  }

  # I have not commented the next four loops for
  # as they are repeating the same code. I apologise for
  # not refactoring them into a function!

  # RUN FOR OVEN
  for (each_app in APPLIANCE[3]) {
    
    cat(paste("Fitting for ", each_app, "\n", sep = ""))
    train_subset_df <- app_train[[each_app]][, c(..each_app, ..BASE_COLUMNS)]

    cat(paste("Fitting GBM "))
    fit_gbm <- train(baseline[[each_app]],
      data = train_subset_df, method = "gbm", metric = metric,
      trControl = ts_CV
    )

    model_list <- list(gbm = fit_gbm)

    pred_valdn_list <- predict(model_list, df_valdn, type = "raw")
    pred_test_list <- predict(model_list, df_test, type = "raw")
    col_names <- paste(each_app, names(model_list), sep = "_")

    for (each_model in names(model_list)) {
      check_f1 <- confusionMatrix(pred_valdn_list[[each_model]], df_valdn[[each_app]], positive = "on")
      cat(paste(each_model, " -> ", names(check_f1$byClass), ": ", check_f1$byClass, "\n", sep = ""))

      run_data <- data.frame(as.list(c(app = each_app, model = each_model, check_f1$byClass)))
      df_results <- rbind(df_results, run_data)
    }

    df_test[, c(col_names) := c(pred_test_list)]
    df_valdn[, c(col_names) := c(pred_valdn_list)]
  }

  # RUN FOR WASH
  for (each_app in APPLIANCE[4]) {
    
    cat(paste("Fitting for ", each_app, "\n", sep = ""))
    train_subset_df <- app_train[[each_app]][, c(..each_app, ..BASE_COLUMNS)]

    cat(paste("Fitting XGB"))
    fit_xgb <- train(baseline[[each_app]],
      data = train_subset_df, method = "xgbTree", metric = metric,
      trControl = ts_CV, maximize = FALSE, scale_pos_weight = 1 # added scalepos and maximize.
    )

    model_list <- list(xgb = fit_xgb)

    pred_valdn_list <- predict(model_list, df_valdn, type = "raw")
    pred_test_list <- predict(model_list, df_test, type = "raw")
    col_names <- paste(each_app, names(model_list), sep = "_")

    for (each_model in names(model_list)) {
      check_f1 <- confusionMatrix(pred_valdn_list[[each_model]], df_valdn[[each_app]], positive = "on")
      cat(paste(each_model, " -> ", names(check_f1$byClass), ": ", check_f1$byClass, "\n", sep = ""))

      run_data <- data.frame(as.list(c(app = each_app, model = each_model, check_f1$byClass)))
      df_results <- rbind(df_results, run_data)
    }

    df_test[, c(col_names) := c(pred_test_list)]
    df_valdn[, c(col_names) := c(pred_valdn_list)]
  }


  # RUN FOR DRYER
  for (each_app in APPLIANCE[5]) {
    cat(paste("Fitting for ", each_app, "\n", sep = ""))
    train_subset_df <- app_train[[each_app]][, c(..each_app, ..BASE_COLUMNS)]

    cat(paste("Fitting GBM"))
    fit_gbm <- train(baseline[[each_app]],
      data = train_subset_df, method = "gbm", metric = metric,
      trControl = ts_CV
    )

    model_list <- list(gbm = fit_gbm)

    pred_valdn_list <- predict(model_list, df_valdn, type = "raw")
    pred_test_list <- predict(model_list, df_test, type = "raw")
    col_names <- paste(each_app, names(model_list), sep = "_")

    for (each_model in names(model_list)) {
      check_f1 <- confusionMatrix(pred_valdn_list[[each_model]], df_valdn[[each_app]], positive = "on")
      cat(paste(each_model, " -> ", names(check_f1$byClass), ": ", check_f1$byClass, "\n", sep = ""))

      run_data <- data.frame(as.list(c(app = each_app, model = each_model, check_f1$byClass)))
      df_results <- rbind(df_results, run_data)
    }

    df_test[, c(col_names) := c(pred_test_list)]
    df_valdn[, c(col_names) := c(pred_valdn_list)]
  }

  # After completing the model runs for each appliance
  # three files are save
  # 1. The confusion matrix results
  # 2. The pred_labels.csv for submission
  # 3. The predictions used for calcualating the confusion matrix.

  setwd(RESULTS_FOLDER)
  fwrite(df_results, paste(job_name, "results.csv", sep = "_"))

  # Take the job name and append to the csv file name 
  prepare_prediction_csv(df_test, paste(job_name, "labels", sep = "_"))
  prepare_prediction_csv(df_valdn, paste(job_name, "valdn", sep = "_"))
  setwd(CWD)
}


# This is where execution takes place
df_results <- create_results_table()
run_job("pred")

dt  <- fread("pred_labels.csv")
names(dt)[1]  <-  'col_index'
fwrite(dt,'pred_labels.csv')
