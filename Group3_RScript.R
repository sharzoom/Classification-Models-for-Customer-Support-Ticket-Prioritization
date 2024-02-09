## -------------------------------------------------------------------------------------------------------
library(ggplot2)
library(tidytext)
library(textdata)
library(dplyr)
library(fastDummies)
library(corrplot)
library(rpart)
library(rpart.plot)
library(randomForest)
library(xgboost)
library(e1071)
library(kernlab)
library(class)
library(caret)
library(vip)
library(doParallel)
library(pROC)


## -------------------------------------------------------------------------------------------------------
setwd("C:\\Vishnu Vardhan\\SUNY Buffalo\\Fall 2023\\Data Analytics and Predictive Modelling\\Project")


## -------------------------------------------------------------------------------------------------------
df <- read.csv("customer_support_tickets.csv")


## -------------------------------------------------------------------------------------------------------
head(df, 2)


## -------------------------------------------------------------------------------------------------------
dim(df)


## -------------------------------------------------------------------------------------------------------
str(df)


## -------------------------------------------------------------------------------------------------------
summary(df)


## -------------------------------------------------------------------------------------------------------
colSums(is.na(df))


## -------------------------------------------------------------------------------------------------------
colSums(df=="")


## -------------------------------------------------------------------------------------------------------
#Feature Engineering:


## -------------------------------------------------------------------------------------------------------
text_data <- df[,c(1, 10)]

tidy_data <- text_data %>% 
    unnest_tokens(word, Ticket.Description)


## -------------------------------------------------------------------------------------------------------
afinn_sentiments <- get_sentiments("afinn")

sentiment_analysis <- tidy_data %>%
  left_join(afinn_sentiments, by = "word") %>%
  group_by(Ticket.ID) %>%
  summarise(sentiment_score = mean(value, na.rm = TRUE))


## -------------------------------------------------------------------------------------------------------
df_sentiments <- as.data.frame(sentiment_analysis)
head(df_sentiments, 5)


## -------------------------------------------------------------------------------------------------------
df1 <- df
df1['Ticket.Description.Sentiment.Score'] <- df_sentiments$sentiment_score


## -------------------------------------------------------------------------------------------------------
df1 <- df1 %>% filter(!is.na(Ticket.Description.Sentiment.Score))


## -------------------------------------------------------------------------------------------------------
df2 <- df1[df1$Ticket.Status == 'Closed',]


## -------------------------------------------------------------------------------------------------------
df2['Ticket.Duration (hours)'] <- round(abs(as.numeric(difftime(strptime(df2[,16], format = "%Y-%m-%d %H:%M:%S"), strptime(df2[,15], format = "%Y-%m-%d %H:%M:%S"), units = 'hours'), units = 'hours')), 2)


## -------------------------------------------------------------------------------------------------------
head(df2['Ticket.Duration (hours)'], 5)


## -------------------------------------------------------------------------------------------------------
strptime(df2[,7], format = "%Y-%m-%d")


## -------------------------------------------------------------------------------------------------------
df_ng <- read.csv('name_gender_dataset.csv')
head(df_ng, 5)


## -------------------------------------------------------------------------------------------------------
df2['Customer.First.Names'] <- sapply(strsplit(df2$Customer.Name, " "), `[`, 1)


## -------------------------------------------------------------------------------------------------------
fetch_gender <- function(name) {
  gender <- df_ng$Gender[df_ng$Name == name]
  if (length(gender) > 0) {
    return(gender[1])
  } else {
    return(NA)
  }
}

for (i in 1:nrow(df2)) {
  if (df2$Customer.Gender[i] == "Other") {
    actual_gender <- fetch_gender(df2$Customer.First.Names[i])
    if (!is.na(actual_gender)) {
        if(actual_gender == 'F'){df2$Customer.Gender[i] <- 'Female'}
        else{df2$Customer.Gender[i] <- 'Male'}
    }
  }
}


## -------------------------------------------------------------------------------------------------------
df3 <- df2[, -c(1, 2, 3, 7, 10, 11, 12, 15, 16, 20)]


## -------------------------------------------------------------------------------------------------------
df3


## -------------------------------------------------------------------------------------------------------
dim(df3)


## -------------------------------------------------------------------------------------------------------
theme_set(theme_gray())
dev.off()

options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Priority')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  expand_limits(y = max(table(df3$Ticket.Priority)) * 1.1) +
  geom_text(stat = 'count', aes(label=..count..), vjust = -0.3) +
  labs(title = 'Ticket Priority Levels', 
       subtitle = 'Number of tickets raised at each priority level.',
       caption = 'Ticket priority levels seem mostly evenly distributed.',
       x = 'Level', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold'), 
        axis.title.x = element_text(size = 13, face = 'bold'), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Customer.Gender')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  expand_limits(y = max(table(df3$Customer.Gender)) * 1.1) +
  geom_text(stat = 'count', aes(label=..count..), vjust = -0.3) +
  labs(title = 'Customer Gender Distribution', 
       subtitle = 'Number of tickets raised by each gender.',
       caption = 'There isn\'t any significant disparity in the number of tickets raised my men and women.',
       x = 'Gender', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold'), 
        axis.title.x = element_text(size = 13, face = 'bold'), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 16, repr.plot.height = 10)

ggplot(df3, aes_string(x = 'Product.Purchased')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  labs(title = 'Purchased Product Distribution', 
       subtitle = 'Number of tickets raised for each product.',
       x = 'Product', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 8, face = 'plain', vjust = 0.5, angle = 90), 
        axis.title.x = element_text(size = 12, face = 'bold'), 
        axis.text.y = element_text(size = 8, face = 'bold'), 
        axis.title.y = element_text(size = 12, face = 'bold'),
        plot.margin = unit(c(0.5, 0.5, 0.5, 0.5), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Type')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  expand_limits(y = max(table(df3$Ticket.Type)) * 1.1) +
  geom_text(stat = 'count', aes(label=..count..), vjust = -0.3) +
  labs(title = 'Ticket Type Distribution', 
       subtitle = 'Number of tickets raised for ticket type.',
       caption = 'The ticket types are mostly evenly distributed.',
       x = 'Ticket Type', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold', vjust = 0.5, angle = 30), 
        axis.title.x = element_text(size = 13, face = 'bold'), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 10)

ggplot(df3, aes_string(x = 'Ticket.Subject')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  expand_limits(y = max(table(df3$Ticket.Subject)) * 1.1) +
  labs(title = 'Ticket Subject Distribution', 
       subtitle = 'Number of tickets raised for ticket subject.',
       caption = 'The ticket subjects are mostly evenly distributed.',
       x = 'Ticket Subject', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'plain', vjust = 0.5, angle = 90), 
        axis.title.x = element_text(size = 13, face = 'bold'), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Channel')) + 
  geom_bar(size = 1, colour = 'black', fill = '#adf7b6') +
  expand_limits(y = max(table(df3$Ticket.Channel)) * 1.1) +
  geom_text(stat = 'count', aes(label=..count..), vjust = -0.3) +
  labs(title = 'Ticket Channel Distribution', 
       subtitle = 'Number of tickets raised in each chennel.',
       caption = 'The ticket channels are mostly evenly distributed.',
       x = 'Ticket Channels', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold', vjust = 0.5, angle = 0), 
        axis.title.x = element_text(size = 13, face = 'bold'), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm")) 


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 16, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Customer.Age')) + 
  geom_histogram(size = 1, colour = 'black', fill = '#adf7b6', bins = 52) +
  labs(title = 'Customer Age Distribution', 
       subtitle = 'Number of tickets raised by customers between ages 18-70.',
       caption = 'It appears that there are far more support tickets raised by customers aged 25 than \nany other age.',
       x = 'Customer Age', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold'), 
        axis.title.x = element_text(size = 13, face = 'bold', ), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Customer.Satisfaction.Rating')) + 
  geom_histogram(size = 1, colour = 'black', fill = '#adf7b6', bins = 5) +
  labs(title = 'Customer Satisfaction Rating Distribution', 
       subtitle = 'Distribution of the feedback ratings given by the customers for the support ticket they raised.',
       caption = 'Customer Satisfaction Ratings seem to be mostly evenly distributed.',
       x = 'Customer Satisfaction Rating', y = 'Count') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold'), 
        axis.title.x = element_text(size = 13, face = 'bold', ), 
        axis.text.y = element_text(size = 10, face = 'bold'), 
        axis.title.y = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes(y = df3[, 10])) + 
  geom_boxplot(fill = 'blue', colour = 'black') + 
  labs(title = 'Distribution of Ticket Duration (in hours)', 
       subtitle = 'Boxplot showing the distribution of ticket durations.',
       caption = 'It appears that the usual response time is between 3-12 hours.',
       x = '', y = 'Ticket Duration (hours)') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes(y = df3[, 9])) + 
  geom_boxplot(fill = 'blue', colour = 'black') + 
  labs(title = 'Ticket Description Sentiment Scores', 
       subtitle = 'Boxplot showing the distribution of sentiment scores extracted from the ticket descriptions.',
       caption = 'It appears that the sentiment scores lie mostly between 0 to 1.2. There are some outliers observed on the lower end of the boxplot.',
       x = '', y = 'Sentiment Scores') +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
#Removing outliers:

#Q1 <- quantile(df3[, 9], 0.25)
#Q3 <- quantile(df3[, 9], 0.75)
#IQR <- Q3 - Q1

# Defining the outlier boundaries
#lower_bound <- Q1 - 1.5 * IQR
#upper_bound <- Q3 + 1.5 * IQR

#df4 <- subset(df3, df3[, 9] >= lower_bound & df3[, 9] <= upper_bound)


## -------------------------------------------------------------------------------------------------------
custom_colors <- c("#ff686b","#ffc09f", "#adf7b6","#ffee93")
tp <- factor(df3$Ticket.Priority, levels = c('Critical', 'High', 'Medium', 'Low'))

options(repr.plot.width = 6, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Customer.Gender', fill = 'Ticket.Priority')) + 
  geom_bar(stat = 'count', position = 'dodge', colour = 'black', size = 0.8) +
  labs(title = 'Ticket Priority vs Customer Gender', 
       subtitle = 'Number of tickets raised at each priority level w.r.t the gender of the customer.',
       caption = "It appears there's a slightly higher chance of women raising a ticket that would be classified as Critical than men.",
       y = 'Count', x = 'Gender') + guides(fill = guide_legend(title = 'Ticket Priority')) + scale_fill_manual(values = custom_colors) +
  theme(axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'), 
        legend.text = element_text(size = 14), legend.title = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.3, 0.2, 0.3, 0.3), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 20, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Product.Purchased', fill = 'Ticket.Priority')) + 
  geom_bar(stat = 'count', position = 'dodge', colour = 'black', size = 0.8) +
  labs(title = 'Ticket Priority vs Product Purchased', 
       subtitle = 'Number of tickets raised at each priority level w.r.t the Product Purchased by the customer.',
       y = 'Count', x = 'Product Purchased') + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'plain', angle = 90), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'), 
        legend.text = element_text(size = 14), legend.title = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Type', fill = 'Ticket.Priority')) + 
  geom_bar(stat = 'count', position = 'dodge', colour = 'black', size = 0.8) +
  labs(title = 'Ticket Priority vs Ticket Type', 
       subtitle = 'Number of tickets raised at each priority level w.r.t the Ticket Type by the customer.',
       y = 'Count', x = 'Ticket Type')  + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'), 
        legend.text = element_text(size = 14), legend.title = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 10)

ggplot(df3, aes_string(x = 'Ticket.Subject', fill = 'Ticket.Priority')) + 
  geom_bar(stat = 'count', position = 'dodge', colour = 'black', size = 0.8) +
  labs(title = 'Ticket Priority vs Ticket Subject', 
       subtitle = 'Number of tickets raised at each priority level w.r.t the Ticket Subject by the customer.',
       y = 'Count', x = 'Ticket Subject')  + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 10, face = 'plain', angle = 90, vjust = 0.7), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'), 
        legend.text = element_text(size = 14), legend.title = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 8)

ggplot(df3, aes_string(x = 'Ticket.Channel', fill = 'Ticket.Priority')) + 
  geom_bar(stat = 'count', position = 'dodge', colour = 'black', size = 0.8) +
  labs(title = 'Ticket Priority vs Ticket Channel', 
       subtitle = 'Number of tickets raised at each priority level w.r.t the Ticket Channel by the customer.',
       caption = 'The tickets are evenly distributed at each priority level w.r.t Ticket Channel.',
       y = 'Count', x = 'Ticket Channel')  + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'), 
        legend.text = element_text(size = 14), legend.title = element_text(size = 13, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Priority', y = 'Customer.Age', fill = 'Ticket.Priority')) + 
  geom_boxplot(size = 0.8, colour = 'black', show.legend = FALSE) + 
  labs(title = 'Ticket Priority vs Customer Age', 
       subtitle = 'Distribution of tickets raised at each priority level w.r.t the age of the customer.',
       caption = 'The tickets are evenly distributed at each priority level across all ages.',
       x = 'Ticket Priority', y = 'Customer Age') + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Priority', y = 'Customer.Satisfaction.Rating', fill = 'Ticket.Priority')) + 
  geom_boxplot(size = 0.8, colour = 'black', show.legend = FALSE) + 
  labs(title = 'Ticket Priority vs Customer Satisfaction Rating', 
       subtitle = 'Distribution of tickets raised at each priority level w.r.t customer satisfaction rating.',
       caption = 'The tickets are evenly distributed at each priority level across all customer satisfaction ratings.',
       x = 'Ticket Priority', y = 'Customer Satisfaction Rating') + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Priority', y = df3[, 10], fill = 'Ticket.Priority')) + 
  geom_boxplot(size = 0.8, colour = 'black', show.legend = FALSE) + 
  labs(title = 'Ticket Priority vs Ticket Resolution Duration', 
       subtitle = 'Distribution of tickets raised at each priority level w.r.t ticket resolution duration.',
       caption = 'It appears that critical tickets take slightly less time to get resolved than other priority levels.',
       x = 'Ticket Priority', y = 'Ticket Resolution Duration') + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 10, repr.plot.height = 6)

ggplot(df3, aes_string(x = 'Ticket.Priority', y = df3[, 9], fill = 'Ticket.Priority')) + 
  geom_boxplot(size = 0.8, colour = 'black', show.legend = FALSE) + 
  labs(title = 'Ticket Priority vs Ticket Description Sentiment Score', 
       subtitle = 'Distribution of tickets raised at each priority level w.r.t ticket description sentiment score.',
       caption = 'It appears that low priority tickets have slightly higher ticket description sentiment score.',
       x = 'Ticket Priority', y = 'Ticket Description Sentiment Score') + guides(fill = guide_legend(title = 'Ticket Priority')) +scale_fill_manual(values = custom_colors) +
  theme(plot.title = element_text(size = 16, face = 'bold'),
        plot.subtitle = element_text(size = 10, face = 'plain'),
        plot.caption = element_text(size = 10, face = 'italic', hjust = 0),
        axis.text.x = element_text(size = 14, face = 'bold'), 
        axis.title.x = element_text(size = 14, face = 'bold'), 
        axis.text.y = element_text(size = 14, face = 'bold'), 
        axis.title.y = element_text(size = 14, face = 'bold'),
        plot.margin = unit(c(0.8, 0.8, 0.8, 0.8), "cm"))


## -------------------------------------------------------------------------------------------------------
df_categorical <- df3[, c(2, 3, 4, 5, 7)]
df_numerical <- df3[, c(1, 8, 9, 10)]
df_target <- df3[, c(6)]


## -------------------------------------------------------------------------------------------------------
df_categorical_dummies <- dummy_cols(df_categorical, remove_first_dummy = TRUE)[,-c(1,2,3,4,5)]


## -------------------------------------------------------------------------------------------------------
df_categorical_dummies


## -------------------------------------------------------------------------------------------------------
Ticket.Priority <- factor(df3$Ticket.Priority, levels = c("Low", "Medium", "High", "Critical"), ordered = TRUE)


## -------------------------------------------------------------------------------------------------------
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
df_numerical_normalized <- as.data.frame(lapply(df_numerical, normalize))


## -------------------------------------------------------------------------------------------------------
head(df_numerical_normalized, 5)


## -------------------------------------------------------------------------------------------------------
df4 <- cbind(df_numerical_normalized, df_categorical_dummies, Ticket.Priority)
head(df4, 5)


## -------------------------------------------------------------------------------------------------------
dim(df4)


## -------------------------------------------------------------------------------------------------------
#Correlation Matrix:
corr <- cor(df4[, -c(69)])
corr


## -------------------------------------------------------------------------------------------------------
#Heatmap of the correlations:
options(repr.plot.width = 24, repr.plot.height = 24)
heatmap(corr, Rowv = NA, Colv = NA, cexRow = 1.8, cexCol = 1.8, margins = c(32, 32))


## -------------------------------------------------------------------------------------------------------
#Checking for multicollinearity:

corr_df <- as.data.frame(corr)
multicollinearity_state <- (corr_df > 0.9) | (corr_df < -0.9)
multicollinearity_state[multicollinearity_state == TRUE] <- 1

options(repr.plot.width = 24, repr.plot.height = 24)
heatmap(multicollinearity_state, Rowv = NA, Colv = NA, cexRow = 1.8, cexCol = 1.8, margins = c(32, 32))


## -------------------------------------------------------------------------------------------------------
set.seed(150)

split <- sample(1:nrow(df4), 0.6*nrow(df4))

train <- df4[split, ]
validationTest <- df4[-split, ]

split <- sample(1:nrow(validationTest), 0.5*nrow(validationTest))

validation <- validationTest[split, ]
test <- validationTest[-split, ]

dim(train)
dim(validation)
dim(test)


## -------------------------------------------------------------------------------------------------------
model_accuracies <- data.frame(Model = character(), TrainingAccuracy = numeric(), ValidationAccuracy = numeric(), stringsAsFactors = FALSE)

update_model_accuracies <- function(df, model_name, training_accuracy, validation_accuracy) {
    df <- rbind(df, data.frame(Model = model_name, TrainingAccuracy = training_accuracy, ValidationAccuracy = validation_accuracy))
    return(df)
}


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# Define the tune grid
tune_grid <- expand.grid(.cp = 0.001)

# Train the model
dtree <- train(x = train[,-c(69)], y = train[, 69], 
               method = "rpart",
               trControl = train_control, 
               tuneGrid = tune_grid)


## -------------------------------------------------------------------------------------------------------
detailed_results <- dtree$resample
print(detailed_results)


## -------------------------------------------------------------------------------------------------------
dt_predictions_train <- predict(dtree, train, type = "raw")
dt_predictions_validation <- predict(dtree, validation, type = "raw")


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(dt_predictions_train, as.factor(train$Ticket.Priority))
conf_matrix_validation <- confusionMatrix(dt_predictions_validation, as.factor(validation$Ticket.Priority))

conf_matrix_table <- conf_matrix_train$table
print("Decision Tree Confusion Matrix (Train Data):")
print(conf_matrix_table)

model_accuracies <- update_model_accuracies(model_accuracies, 'Decision Tree', conf_matrix_train$overall[['Accuracy']], conf_matrix_validation$overall[['Accuracy']])
model_accuracies


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

# Train the model
rf <- train(x = train[,-c(69)], y = train$Ticket.Priority, 
               method = "rf",
               trControl = train_control)


## -------------------------------------------------------------------------------------------------------
detailed_results <- rf$resample
print(detailed_results)


## -------------------------------------------------------------------------------------------------------
rf_predictions_train <- predict(rf, train, type = "raw")
rf_predictions_validation <- predict(rf, validation, type = "raw")


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(rf_predictions_train, train$Ticket.Priority)
conf_matrix_validation <- confusionMatrix(rf_predictions_validation, validation$Ticket.Priority)

conf_matrix_table <- conf_matrix_train$table
print("Random Forest Confusion Matrix (Train Data):")
print(conf_matrix_table)

model_accuracies <- update_model_accuracies(model_accuracies, 'Random Forest', conf_matrix_train$overall[['Accuracy']], conf_matrix_validation$overall[['Accuracy']])
model_accuracies


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

xg <- train(x = train[,-c(69)], y = as.factor(train$Ticket.Priority), method = "xgbTree", trControl = train_control)


## -------------------------------------------------------------------------------------------------------
detailed_results <- xg$resample
print(detailed_results)


## -------------------------------------------------------------------------------------------------------
xg_predictions_train <- predict(xg, train, type = "raw")
xg_predictions_validation <- predict(xg, validation, type = "raw")


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(xg_predictions_train, train$Ticket.Priority)
conf_matrix_validation <- confusionMatrix(xg_predictions_validation, validation$Ticket.Priority)

conf_matrix_table <- conf_matrix_train$table
print("XGBoost Confusion Matrix (Train Data):")
print(conf_matrix_table)

model_accuracies <- update_model_accuracies(model_accuracies, 'eXtreme Gradient Boosting (XGBoost)', conf_matrix_train$overall[['Accuracy']], conf_matrix_validation$overall[['Accuracy']])
model_accuracies


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

svm <- train(Ticket.Priority ~ ., train, method = "svmPoly", trControl = train_control)


## -------------------------------------------------------------------------------------------------------
detailed_results <- svm$resample
print(detailed_results)


## -------------------------------------------------------------------------------------------------------
svm_predictions_train <- predict(svm, train, type = "raw")
svm_predictions_validation <- predict(svm, validation, type = "raw")


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(svm_predictions_train, train$Ticket.Priority)
conf_matrix_validation <- confusionMatrix(svm_predictions_validation, validation$Ticket.Priority)

conf_matrix_table <- conf_matrix_train$table
print("SVM Confusion Matrix (Train Data):")
print(conf_matrix_table)

model_accuracies <- update_model_accuracies(model_accuracies, 'Support Vector Machine (SVM)', conf_matrix_train$overall[['Accuracy']], conf_matrix_validation$overall[['Accuracy']])
model_accuracies


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

tune_grid <- expand.grid(k = 5)

knn <- train(x = train[,-c(69)], y = train$Ticket.Priority, method = "knn", 
               trControl = train_control, tuneGrid = tune_grid)


## -------------------------------------------------------------------------------------------------------
detailed_results <- knn$resample
print(detailed_results)


## -------------------------------------------------------------------------------------------------------
knn_predictions_train <- predict(knn, train, type = "raw")
knn_predictions_validation <- predict(knn, validation, type = "raw")


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(knn_predictions_train, train$Ticket.Priority)
conf_matrix_validation <- confusionMatrix(knn_predictions_validation, validation$Ticket.Priority)

conf_matrix_table <- conf_matrix_train$table
print("KNN Confusion Matrix (Train Data):")
print(conf_matrix_table)

model_accuracies <- update_model_accuracies(model_accuracies, 'K-Nearest Neighbors (KNN)', conf_matrix_train$overall[['Accuracy']], conf_matrix_validation$overall[['Accuracy']])
model_accuracies


## -------------------------------------------------------------------------------------------------------
#Final Model:


## -------------------------------------------------------------------------------------------------------
VI <- vi(rf)
VI


## -------------------------------------------------------------------------------------------------------
options(repr.plot.width = 18, repr.plot.height = 12)
ggplot(VI, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance", x = "Variables", y = "Importance")


## -------------------------------------------------------------------------------------------------------
impVariables <- VI$Variable[VI$Importance > 50]
impVariables


## -------------------------------------------------------------------------------------------------------
train_control <- trainControl(method = "cv", number = 5, verboseIter = TRUE)

tune_grid <- expand.grid(nrounds = c(100, 150),
                         max_depth = c(3, 5, 7),
                         eta = c(0.01, 0.1),
                         gamma = c(0, 0.1),
                         colsample_bytree = c(0.5, 0.7),
                         min_child_weight = c(1, 3),
                         subsample = c(0.6, 0.7, 0.8))


## -------------------------------------------------------------------------------------------------------
final_model <- train(x = train[, impVariables], y = train$Ticket.Priority, method = "xgbTree", trControl = train_control, tuneGrid = tune_grid)


## -------------------------------------------------------------------------------------------------------
print(final_model)


## -------------------------------------------------------------------------------------------------------
final_model$results


## -------------------------------------------------------------------------------------------------------
final_model$resample


## -------------------------------------------------------------------------------------------------------
final_model$finalModel


## -------------------------------------------------------------------------------------------------------
xgFinal_predictions_train <- predict(final_model, train[, impVariables])
xgFinal_predictions_validation <- predict(final_model, validation[, impVariables])
xgFinal_predictions_test <- predict(final_model, test[, impVariables])


## -------------------------------------------------------------------------------------------------------
conf_matrix_train <- confusionMatrix(xgFinal_predictions_train, as.factor(train$Ticket.Priority))
conf_matrix_validation <- confusionMatrix(xgFinal_predictions_validation, as.factor(validation$Ticket.Priority))
conf_matrix_test <- confusionMatrix(xgFinal_predictions_test, as.factor(test$Ticket.Priority))


## -------------------------------------------------------------------------------------------------------
conf_matrix_table <- conf_matrix_train$table
print("XGBoost Final Model Confusion Matrix (Train Data):")
print(conf_matrix_table)
accuracy <- conf_matrix_train$overall[['Accuracy']]
precision <- conf_matrix_train$byClass[, 'Precision']
recall <- conf_matrix_train$byClass[, 'Recall']
F1 <- conf_matrix_train$byClass[, 'F1']
train_metrics <- c(Accuracy = accuracy, Precision = mean(precision, na.rm = TRUE), Recall = mean(recall, na.rm = TRUE), F1_Score = mean(F1, na.rm = TRUE))


## -------------------------------------------------------------------------------------------------------
conf_matrix_table <- conf_matrix_validation$table
print("XGBoost Final Model Confusion Matrix (Validation Data):")
print(conf_matrix_table)
accuracy <- conf_matrix_validation$overall[['Accuracy']]
precision <- conf_matrix_validation$byClass[, 'Precision']
recall <- conf_matrix_validation$byClass[, 'Recall']
F1 <- conf_matrix_validation$byClass[, 'F1']
validation_metrics <- c(Accuracy = accuracy, Precision = mean(precision, na.rm = TRUE), Recall = mean(recall, na.rm = TRUE), F1_Score = mean(F1, na.rm = TRUE))


## -------------------------------------------------------------------------------------------------------
conf_matrix_table <- conf_matrix_test$table
print("XGBoost Final Model Confusion Matrix (Test Data):")
print(conf_matrix_table)
accuracy <- conf_matrix_test$overall[['Accuracy']]
precision <- conf_matrix_test$byClass[, 'Precision']
recall <- conf_matrix_test$byClass[, 'Recall']
F1 <- conf_matrix_test$byClass[, 'F1']
test_metrics <- c(Accuracy = accuracy, Precision = mean(precision, na.rm = TRUE), Recall = mean(recall, na.rm = TRUE), F1_Score = mean(F1, na.rm = TRUE))


## -------------------------------------------------------------------------------------------------------
results <- data.frame(
  Train = train_metrics,
  Validation = validation_metrics,
  Test = test_metrics
)


## -------------------------------------------------------------------------------------------------------
print(results)

