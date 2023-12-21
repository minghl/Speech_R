######################################################################################
# Decision Tree, Conditional Inference Tree, and Random Forest classifiers tutorial  #
#                                                                                    #
# Written by Alessandro De Luca on 12.10.2023                                        #
# for 23HS - Computational Processing of Speech Rhythm for Language and              #
#            Speaker Classification.                                                 #
# Copyright: CC0 No rights reserved.                                                 #
######################################################################################

install.packages("pROC")


# Libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)        # pre-processing
library(rpart)        # decision tree
library(partykit)     # conditional inference tree
library(randomForest) # random forest classifier
library(pROC)         # ROC and AUC

newCv <- CV_measures_2[c("speaker", "sentence","rPVI_Con_tier2","rPVI_C_tier3","rPVI_CV_tier3","percentV_tier3","nPVI_Vof_tier2","nPVI_V_tier3","nPVI_CV_tier3")]

dd <- merge(intensityVariability,newCv, by.x = c("speakerId", "sentence"), by.y = c("speaker", "sentence"))
merged_df
dd <- dd %>% select(-speakerId, -sentence,-sex)
# cart90 dataset
print(dim(merged_df))
colnames(merged_df)

# selection of features
dd <- select(car90, -c(Country, Model2, Tires)) # filter only variable of interest
dd$Trans1 <- as.factor(ifelse(dd$Trans1=="", "no", dd$Trans1))
dd$Trans2 <- as.factor(ifelse(dd$Trans2=="", "no", dd$Trans2))
dd <- select(dd, Type, everything())     # move the response var to first col
dd <- dd %>% drop_na(agegroup)               # drop where Type unknown

# Preprocessing
set.seed(89) # seed

# Data Partitioning
index <- createDataPartition(dd$agegroup, p = 0.8, list = FALSE)
train <- dd[index, ]
test <- dd[-index, ]

# distribution of classes
dd %>%
  group_by(agegroup) %>%
  summarise(n = n(), prop = n() / nrow(dd))

# Class balancing
# undersampling
colnames(train)
train$agegroup <- factor(train$agegroup)
train_downsamp <- downSample(
  x=train[, 2:16], y=train$agegroup, list=FALSE, yname="agegroup"
)


train_downsamp %>%
  group_by(agegroup) %>%
  summarise(n = n(), prop = n() / nrow(train))

# oversampling
train_upsamp <- upSample(
  x=train[, 2:31], y=train$Type, list=FALSE, yname="Type"
)
# move Type again to the front of the dataframe
train_upsamp <- train_upsamp %>% select(Type, everything())

train_upsamp %>%
  group_by(Type) %>%
  summarise(n = n(), prop = n() / nrow(train_upsamp))


# Classic Decision Tree
dec.tree <- rpart(
  formula = agegroup ~ ., data = train_downsamp, method = "class" 
)
printcp(dec.tree) # display the results
summary(dec.tree) # summary of splits

# plotting the tree
plot(dec.tree, uniform = TRUE, main = "Classification Tree for Type")
text(dec.tree, use.n = TRUE, all=TRUE, cex=.8)


# Conditional Inference Tree
ct <- ctree(
  formula = agegroup ~ ., data = train_downsamp, 
)
plot(ct, main="Conditional Inference Tree for Type")

# test statistics per node
sctest.constparty(ct)

# simpler trees
simple.ct <- ctree(
  formula = agegroup ~ ., data = train_downsamp, control = ctree_control(maxdepth = 2)
)
plot(simple.ct)


# Random forest classifier
rf <- randomForest(
  x = train_downsamp[, 2:16], y=train_downsamp$agegroup, 
  maxnodes = 10, ntree = 20, importance = TRUE
) # ERROR (NAs in predictors)

train.imputed <- rfImpute(
  x = train_downsamp[, 2:16], y=train_downsamp[, 1], ntree = 10,
) # Impute the missing values during the training
# rename y
colnames(train.imputed)[1] <- "agegroup"
rf <- randomForest(
  x = train[, 2:16], y = train$agegroup,
  ntree = 20, maxnodes = 10, importance = TRUE
)
rf

# Variable importance
varImpPlot(rf, main="Feature importance")


# Compare the models...
test.features <- test[, 2:16]
test.features <- na.roughfix(test.features)

pred.dec.tree <- predict(dec.tree, newdata = test.features, type="class")
pred.ct <- predict(ct, newdata = test.features, type="response")
pred.ct.simple <- predict(simple.ct, newdata = test.features, type="response")
pred.rf <- predict(rf, newdata = test.features, type="class")


levels(test$agegroup)
levels(pred.dec.tree)
all_levels <- c("o", "y")
data_test <- factor(test$agegroup, levels = all_levels)
acc.dec.tree <- confusionMatrix(pred.dec.tree, data_test)$overall
acc.ct <- confusionMatrix(pred.ct, data_test)$overall
acc.simple.ct <- confusionMatrix(pred.ct.simple, data_test)$overall
acc.rf <- confusionMatrix(pred.rf, data_test)$overall

compare_df <- data.frame(
  model = c("rpart", "ctree", "simple ctree","randomForest"),
  accuracy = rbind(acc.dec.tree[[1]], acc.ct[[1]], acc.simple.ct[[1]], acc.rf[[1]])
)
arrange(compare_df, accuracy)
confusionMatrix(pred.dec.tree, data_test)
confusionMatrix(pred.ct, data_test)
confusionMatrix(pred.ct.simple, data_test)
confusionMatrix(pred.rf, data_test)
# A nicer ROC plot
pred.probs <- predict(rf, newdata = na.roughfix(test[,2:16]), type="prob")
# Conditional probabilities for each observation and each class

# Focusing on the level: Medium
pcompact <- pred.probs[, 1]

# ROC
r <- multiclass.roc(test$agegroup, pcompact, percent=TRUE)
print(r)

# focusing only on Compact
r1 <- r[['rocs']][[1]]
# Plot
plot.roc(
  r1, print.auc=TRUE, 
  auc.polygon=TRUE,
  grid=c(0.1, 0.2),
  grid.col=c("green", "red"),
  max.auc.polygon=TRUE,
  auc.polygon.col="lightblue",
  print.thres=TRUE,
  main= 'ROC Curve'
)
