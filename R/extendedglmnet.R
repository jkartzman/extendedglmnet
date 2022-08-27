#' Extended Generalized Linear Models
#'
#' This function allows you to train a linear model using several
#' different algorithms, including lasso, ridge, and random lasso.  It
#' also allows you to use two different types of response variables,
#' logistic and linear.
#' @param trainX Input matrix of dimension (n observations) x (n variables) where each row is an observation.  Number of parameters must be 2 or more
#' @param trainY Response variable containing a vector of n observations.  Elements should be continuous for response="linear" and should contain 0 or 1 for response="logistic"
#' @param response String representing type of response variable.  Must be either "linear" or "logistic".  Defaults to "linear"
#' @param model String representing model type.  Must be "lasso", "ridge", or "random lasso".  Defaults to "lasso"
#' @param B Integer representing number of samples drawn in random lasso.  Defaults to 500
#' @param q1 Integer representing number of parameters sampled when measuring importance in random lasso.
#' @param q2 Integer representing number of parameters sampled when calculating final parameters in random lasso.
#' @export
extendedglmnet <- function(trainX, trainY, response=c("linear","logistic"), model=c("lasso","ridge","random lasso"), B=500, q1, q2){
  np <- dim(trainX)
  if (np[2] <= 1){
    stop("x should be a matrix with 2 or more columns")
  }
  if (q1 < 2 || q1 > ncol(trainX)){
    stop("q1 must be greater than 1 and less than or equal to number of parameters")
  }
  if (q2 < 2 || q2 > ncol(trainX)){
    stop("q2 must be greater than 1 and less than or equal to number of parameters")
  }
  response <- match.arg(response)
  model <- match.arg(model)
  if(model=="ridge"){
    return(ridge(trainX,trainY,response=response))
  }
  else if(model=="lasso"){
    return(lasso(trainX,trainY,response=response))
  }
  else if(model=="random lasso"){
    return(random.lasso(trainX,trainY,response=response,B=B,q1=q1,q2=q2))
  }
}

lasso <- function(trainX, trainY, response){
  if(response == "linear"){
    cv.lasso <- cv.glmnet(trainX, trainY, alpha=1, nfolds=5)
    model <- glmnet(trainX, trainY, alpha=1, lambda=cv.lasso$lambda.min)
  }
  if(response == "logistic"){
    cv.lasso <- cv.glmnet(trainX, trainY, alpha=1, nfolds=5, family="binomial")
    model <- glmnet(trainX, trainY, alpha=1, family="binomial", lambda=cv.lasso$lambda.min)
  }
  return(model)
}

ridge <- function(trainX, trainY, response){
  if(response == "linear"){
    cv.ridge <- cv.glmnet(trainX, trainY, alpha=0, nfolds=5)
    model <- glmnet(trainX, trainY, alpha=0, lambda=cv.ridge$lambda.min)
  }
  if(response == "logistic"){
    cv.ridge <- cv.glmnet(trainX, trainY, alpha=0, nfolds=5, family="binomial")
    model <- glmnet(trainX, trainY, alpha=0, family="binomial", lambda=cv.ridge$lambda.min)
  }
  return(model)
}

random.lasso <- function(trainX, trainY, response, B, q1, q2){
  sample.rows <- sample(nrow(trainX), B*nrow(trainX), replace = TRUE)
  sample.X <- trainX[sample.rows, ]
  sample.Y <- trainY[sample.rows]
  importance.measures <- rep(0,ncol(trainX))
  n <- nrow(trainX)
  if(response == "linear"){
    for(b in 1:B){
      sample.parameters <- sample(ncol(trainX),q1)
      bth.sample.X <- sample.X[((b-1)*n+1):(b*n), sample.parameters]
      bth.sample.Y <- sample.Y[((b-1)*n+1):(b*n)]
      cv.lasso <- cv.glmnet(bth.sample.X, bth.sample.Y, alpha=1, nfolds=5)
      model <- glmnet(bth.sample.X, bth.sample.Y, alpha=1, lambda=cv.lasso$lambda.min)
      importance.measures[sample.parameters] <- importance.measures[sample.parameters]+as.vector(coef(model))[2:length(as.vector(coef(model)))]
    }
    importance.measures <- abs(importance.measures / B)
    selection.prob <- importance.measures/sum(importance.measures)
    selected.coef <- rep(0,ncol(trainX)+1)
    for(b in 1:B){
      sample.parameters <- sample(ncol(trainX),q2,prob = selection.prob)
      bth.sample.X <- sample.X[((b-1)*n+1):(b*n), sample.parameters]
      bth.sample.Y <- sample.Y[((b-1)*n+1):(b*n)]
      cv.lasso <- cv.glmnet(bth.sample.X, bth.sample.Y, alpha=1, nfolds=5)
      model <- glmnet(bth.sample.X, bth.sample.Y, alpha=1, lambda=cv.lasso$lambda.min)
      selected.coef[sample.parameters+1] <- selected.coef[sample.parameters+1]+as.vector(coef(model))[2:length(as.vector(coef(model)))]
      selected.coef[1] <- selected.coef[1]+as.vector(coef(model))[1]
    }
    final.coef <- selected.coef/B
  }
  if(response == "logistic"){
    for(b in 1:B){
      sample.parameters <- sample(ncol(trainX),q1)
      bth.sample.X <- sample.X[((b-1)*n+1):(b*n), sample.parameters]
      bth.sample.Y <- sample.Y[((b-1)*n+1):(b*n)]
      cv.lasso <- cv.glmnet(bth.sample.X, bth.sample.Y, alpha=1, nfolds=5, family="binomial")
      model <- glmnet(bth.sample.X, bth.sample.Y, alpha=1, lambda=cv.lasso$lambda.min, family="binomial")
      importance.measures[sample.parameters] <- importance.measures[sample.parameters]+as.vector(coef(model))[2:length(as.vector(coef(model)))]
    }
    importance.measures <- abs(importance.measures / B)
    selection.prob <- importance.measures/sum(importance.measures)
    selected.coef <- rep(0,ncol(trainX)+1)
    for(b in 1:B){
      sample.parameters <- sample(ncol(trainX),q2,prob = selection.prob)
      bth.sample.X <- sample.X[((b-1)*n+1):(b*n), sample.parameters]
      bth.sample.Y <- sample.Y[((b-1)*n+1):(b*n)]
      cv.lasso <- cv.glmnet(bth.sample.X, bth.sample.Y, alpha=1, nfolds=5, family="binomial")
      model <- glmnet(bth.sample.X, bth.sample.Y, alpha=1, lambda=cv.lasso$lambda.min, family="binomial")
      selected.coef[sample.parameters+1] <- selected.coef[sample.parameters+1]+as.vector(coef(model))[2:length(as.vector(coef(model)))]
      selected.coef[1] <- selected.coef[1]+as.vector(coef(model))[1]
    }
    final.coef <- selected.coef/B
  }
  return(final.coef)
}




