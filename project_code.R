# NAME: Sadou Barry | NUMBER: 8226
# Project: OKC Analyst Intern Project - NBA Shot Outcome Prediction Model

library(readr)
library(xgboost)

# 1. Load Datasets
train_df <- read_csv("training.csv.gz")
test_df  <- read_csv("testing.csv.gz")

train_df <- as.data.frame(train_df)
test_df  <- as.data.frame(test_df)

# 2. Evaluation Metric (Log-Loss)
calc_log_loss <- function(y_true, y_pred, eps = 1e-15) {
  y_pred <- pmin(pmax(y_pred, eps), 1 - eps)
  -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
}

# 3. Basketball Informed Feature Engineering
prepare_features <- function(df) {
  if("outcome" %in% names(df)) {
    df$target <- as.numeric(df$outcome)
  }
  
  # Core Spatial Geometry
  # Rim center coordinate: (-41.75, 0)
  df$dist_to_rim <- sqrt((df$locationx - (-41.75))^2 + (df$locationy - 0)^2)
  df$shot_angle  <- atan2(df$locationy, df$locationx + 41.75)
  
  # Basketball Domain Features
  df$is_corner_three     <- as.numeric(df$three & abs(df$locationy) > 22 & df$locationx > -30)
  df$is_late_clock       <- as.numeric(df$shotclock <= 4)
  df$def_pressure_ratio  <- df$closestdefdist / (df$dist_to_rim + 1)
  
  # Custom Features
  df$dist_to_rim_sq   <- df$dist_to_rim^2
  df$is_moving_jumper <- as.numeric(df$shottype == "jumper" & df$shooterspeed > 5)
  
  # Flags & Categoricals
  df$three_flag     <- as.numeric(df$three)
  df$contested_flag <- as.numeric(df$contested)
  df$shottype       <- as.factor(df$shottype)
  df$gamestate      <- as.factor(df$gamestate)
  df$month          <- as.factor(df$month)
  
  return(df)
}

train_df <- prepare_features(train_df)
test_df  <- prepare_features(test_df)

# 4. Train / Validation Split (80/20)
set.seed(30)
train_idx <- sample(seq_len(nrow(train_df)), size = 0.8 * nrow(train_df))

val_set   <- train_df[-train_idx, ]
train_set <- train_df[train_idx, ]

# 5. Baseline Logistic Regression
model_formula <- target ~ dist_to_rim + dist_to_rim_sq + shot_angle + locationx + locationy + 
  shotclock + is_late_clock + dribblesbefore + 
  closestdefdist + def_pressure_ratio + shooterspeed + is_moving_jumper + 
  num_contesters + contested_flag + three_flag + is_corner_three + 
  shottype + gamestate

logistic_model <- glm(model_formula, data = train_set, family = binomial)
val_preds_log  <- predict(logistic_model, newdata = val_set, type = "response")
log_loss_base  <- calc_log_loss(val_set$target, val_preds_log)

cat("Validation Log-Loss (Baseline Logistic Regression):", round(log_loss_base, 5), "\n")

# 6. Advanced XGBoost Model
feature_cols <- c("dist_to_rim", "dist_to_rim_sq", "shot_angle", "locationx", "locationy", 
                  "shotclock", "is_late_clock", "dribblesbefore", 
                  "closestdefdist", "def_pressure_ratio", "shooterspeed", "is_moving_jumper", 
                  "num_contesters", "contested_flag", "three_flag", "is_corner_three")

train_set$shottype_num  <- as.numeric(train_set$shottype)
val_set$shottype_num    <- as.numeric(val_set$shottype)
test_df$shottype_num    <- as.numeric(test_df$shottype)

train_set$gamestate_num <- as.numeric(train_set$gamestate)
val_set$gamestate_num   <- as.numeric(val_set$gamestate)
test_df$gamestate_num   <- as.numeric(test_df$gamestate)

all_feature_cols <- c(feature_cols, "shottype_num", "gamestate_num")

dtrain <- xgb.DMatrix(data = as.matrix(train_set[, all_feature_cols]), label = train_set$target)
dval   <- xgb.DMatrix(data = as.matrix(val_set[, all_feature_cols]), label = val_set$target)
dtest  <- xgb.DMatrix(data = as.matrix(test_df[, all_feature_cols]))

params <- list(
  booster          = "gbtree",
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = 0.05,
  max_depth        = 6,
  subsample        = 0.8,
  colsample_bytree = 0.8
)

set.seed(30)
xgb_model <- xgb.train(
  params                = params,
  data                  = dtrain,
  nrounds               = 500,
  evals                 = list(val = dval),
  early_stopping_rounds = 20,
  verbose               = 0
)

xgb_val_preds <- predict(xgb_model, dval)
xgb_log_loss  <- calc_log_loss(val_set$target, xgb_val_preds)

cat("Validation Log-Loss (XGBoost Final):", round(xgb_log_loss, 5), "\n")

# 7. Generate & Export Final Predictions
submission <- read.csv("submission.csv")
submission$make_prob <- predict(xgb_model, dtest)
write.csv(submission, "submission.csv", row.names = FALSE)

cat("\n--- SUMMARY ---")
cat("\nPredictions summary check:\n")
print(summary(submission$make_prob))
cat("\nsubmission.csv successfully written!\n")