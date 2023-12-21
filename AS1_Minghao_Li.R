# Libraries
library(dplyr)
library(ggplot2)
library(RCurl)   # get data from the internet
library(caret)   # pre-processing
library(MASS)    # LDA
library(ggord)   # LDA biplot
library(dlookr)  # normality test
library(rstatix) # for Box's M statistic
install.packages("ggord")

set.seed(123)  # 你可以使用任何数字作为种子
index <- sample(1:nrow(Book2), nrow(Book2) * 0.8)
train_set <- Book2[index, ]
test_set <- Book2[-index, ]
train_response <- subset(train_set)$dialect
test_response <- subset(test_set)$dialect
to.exclude <- c("file", "speaker","gender","dialect","sentence_number") # non-feature columns
feature_test <- test_set[, !colnames(test_set) %in% to.exclude] # only features
feature_train <- train_set[, !colnames(train_set) %in% to.exclude] # only features
feature_Book2 <- Book2[, !colnames(Book2) %in% to.exclude] # only features


preproc.params <- feature_train %>%
  preProcess(method = c("center", "scale"))
# Standardizing
train <- preproc.params %>% predict(feature_train)
test <- preproc.params %>% predict(feature_test) # using the same parameters
feature <- preproc.params %>% predict(feature_Book2) # using the same parameters

box_m(data = feature_Book2, group = Book2$dialect)

Book2 %>%
  group_by(dialect) %>%
  normality(.) %>%
  filter(p_value > 0.01) %>%
  arrange(abs(p_value)) %>%
  print(n = 100) # or use Inf for all rows

library(cowplot) # for plot_grid

plot_list <- list()
for (feature in names(Book2[1:10])) {
  p <- ggplot(data = Book2, mapping = aes(x = feature, fill = dialect)) +
    geom_density(alpha = .3) +
    labs(title = paste("Density of", feature)) +
    theme_bw()
  plot_list[[feature]] <- p
}
plot_grid(plotlist = plot_list)

lda.model <- lda(train, grouping = train_response)
str(lda.model)
# training predictions
train.lda <- predict(lda.model, train) # transformed data in train.lda$x

# Visualizations:
# scatter plot
# Create a data frame to use with ggplot2
plot.dd <- data.frame(train.lda$x, "phoneme" = train_response)
ggplot(data = plot.dd, mapping = aes(x = LD1, y = LD2, color = phoneme)) +
  geom_point(alpha = .4)

# with ellipses
ggord(lda.model, grp_in = train_response, arrow = NULL, txt = NULL)
# I set `arrow = NULL` and `txt = NULL` to suppress the plotting of arrows and 
# labels for the original variables (it would be 256 arrows!). If you have less 
# features you can leave them to their default values to be able to see the 
# directions of the vectors representing the original features in the reduced LDA space.

# Pairs plot
plot(lda.model)

# histograms of the first discriminant function
par(mar = c(1, 1, 1, 1))
plot(lda.model, dimen=1)

# Evaluation
test.pred <- predict(lda.model, test)
train.pred <- predict(lda.model, train)
test.acc <- length(which(test.pred$class == test_response)) / length(test_response)
train.acc <- length(which(train.pred$class == train_response)) / length(train_response)
print(paste("Training accuracy =", train.acc))
print(paste("Testing accuracy =", test.acc))
library(ggplot2)
library(lattice)
library(caret)
predictions <- factor(test.pred$class, levels = c("zh", "ba", "be"))
actual <- factor(test_response, levels = c("zh", "ba", "be"))
predictions <- predict(lda.model, test)
confusionMatrix(predictions$class, actual)


