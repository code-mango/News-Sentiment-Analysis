# News Sentiment Analysis
This is a large data set of news items and their respective social feedback on multiple platforms: Facebook, Google+ and LinkedIn.
The collected data relates to a period of 8 months, between November 2015 and July 2016, accounting for about 100,000 news items on four different topics: economy, microsoft, obama and palestine.
This data set is tailored for evaluative comparisons in predictive analytics tasks, although allowing for tasks in other research areas such as topic detection and tracking, sentiment analysis in short text, first story detection or news recommendation.

# 1. Dataset Description
Dataset can be downloaded from link:
https://archive.ics.uci.edu/ml/machine-learning-databases/00432/Data/

IDLink (numeric): 	Unique identifier of news items
Title (string):	Title of the news item according to the official media sources
Headline (string):	Headline of the news item according to the official media sources
Source (string):	Original news outlet that published the news item
Topic (string):	Query topic used to obtain the items in the official media sources
PublishDate (timestamp):	Date and time of the news items' publication
SentimentTitle (numeric):	Sentiment score of the text in the news items' title
SentimentHeadline (numeric):	Sentiment score of the text in the news items' headline
Facebook (numeric):	Final value of the news items' popularity according to Facebook data
GooglePlus (numeric):	Final value of the news items' popularity according to Google+
LinkedIn (numeric):	Final value of the news items' popularity according to LinkedIn 


# 2.	Preprocessing Steps

2.1	Data Procuring
•	Importing Libraries for data pre-processing and cleaning (Pandas, Numpy), for visualization (Matplotlib, Seaborn) and for importing the dataset from Gdrive into Google Colab (Google.Colab)
•	Loading the dataset into GoogleColab document using Pandas after importing the dataset from Cloud- Google Drive

2.2	Data Cleaning
•	Original Encoder = Converting the string format of “Topic” to categorical integers. ‘Obama’, ‘Economy’, ‘Palestine’, and ‘Microsoft’ will be 1,2,3,4 respectively.
•	“Headline”, “Source” column using Data.info and .isnull().sum() gives 15 and 279 null values in the dataset
•	Dropping these columns as they don’t add much value to the dataset and hence not required for computational purposes.

2.3	Data Imputation:
 It is the substitution of mean values for missing values
•	SimpleImputer is used to replace the NaN values with the mean values for the columns ‘Topic’, ‘SentimentTitle’, ‘SentimentHeadline’

# 3.	Feature Selection
•	Calculating the outlier values of those below 25% and above 75% on ‘SentimentTitle’, ‘SentimentHeadline’ and removing them.
•	Plotting these values on clustered bar graph through matplotlib library.
•	Trying to find all the correlation with ‘Topic’, ‘SentimentTitle’, and ‘SentimentHeadline’ and then plotting them by graph using Seaborn library.
•	This throws an error as there is no direct correlation between some parameters.
•	There is no collinearity between independent features SentimentTitle, SentimentHeadline, and Topic
•	We can see that the correlation between dependent features Facebook, LinkedIn, Google+ is very near about 0.01-0.5, which shows that there is no evident collinearity between dependent features.

# 4.	Splitting the Dataset
Using the Train_test_split model splitting the dataset into 70:30 ratio.

# 5.	Model Fitting
Applying different models and comparing them to see which one gives the best performance.
Random Forest: Random forest is a bagging method. Random woods have trees that grow in parallel. During the construction of the trees, there is no interaction between them.
•	N_estimators = 100, means 100 number of trees in the forest.
•	Random_state = 0, is used for controlling the randomness while selecting the features for best split at each nodes 

Decision Tree: In the shape of a tree structure, a decision tree constructs regression or classification models. It incrementally cuts down a dataset into smaller and smaller sections while also developing an associated decision tree. A tree with decision nodes and leaf nodes is the end result.
•	Random_state = 0, implies the best split of features at the node which changes with different runs

LightGBM: LightGBM, or Light Gradient Boosting Machine, is a framework for distributed gradient boosting. It is a fast algorithm because it uses Histogram-based splitting, Gradient-based One-Side Sampling (GOSS), and Exclusive Feature Bundling (EFB).
•	GOSS excludes a large part of dataset that contains smaller magnitudes. It will compute on the larger values for estimation. This will help in giving accurate results.
•	EFB will bundle the features and will treat them as a single feature. This improves the speed of learning.

# 6.	Metrics Evaluation
•	Mean Absolute Error 
It is the mean value of absolute sum of difference  of the actual value and the predicted values.
MAE = Absolute Sum [ Actual value – Predicted Value] / Total number of values
•	Mean Absolute Squared Error
It is the squared function of the MAE.
MSE = Squared Mean MAE = MAE*MAE/ Total number of values
•	R Squared Error
Root Mean Squared error is the mean over the square root of MSE.
RMSE = Mean square root of MSE/ Total number of values

Evaluation Tools:
Mean Absolute Error	
Mean Squared Error	
R Squared Error	

# 7.	Conclusion
From Loading the datatsets to cleaning, we applied feature selection and modelling on our dataset. We tried to obtain an accurate score for our dependent variables.
So, from the performance evaluation, we can conclude that LigthGBM gives the highest accuracy as it has the lowest RMSE values among all three models. We discovered that the stages such as data preparation and Feature engineering play an important role in improving the model accuracy and performance.
