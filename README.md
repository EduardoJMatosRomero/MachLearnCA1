# Project Motivation

Ireland is one of the leading countries in the aviation industry, with companies such as Ryanair or Air Lingus being examples of hugely successful companies across Europe, so to base a project on any of the examples of domains in this industry as part of a portfolio of projects seems to me to be hugely valuable.

# Problem Domain

The topic that has been chosen to be analysed is the price of flight ticket in India for domestic flights, considering for the analysis a data collection from 12 national airlines, 5 sources and 6 destinations.

To analyse and making a prediction of the price of tickets it has been analysed 10 features.

- Features
  - Airline
  - Date of journey
  - Source
  - Destination
  - Route
  - Departure time
  - Arrival time
  - Duration
  - Total stops
  - Additional information

# Methods

Since the objective of our project is to develop a predictive model using machine learning and Python, the method that has been decided to be applied is a regression.

1. To develop a regression model, 7 different regression algorithms were tested.
- Linear Regression
- Ridge Regression
- Lasso Regression
- **Decision Tree Regressor**
- **Random Forest Regressor**
- Support Vector Regressor
- K-Nearest Neighbours Regressor

2. The regression models were tested using 3 different approaches of data train-test splitting, with the intention of choosing the train-test splitting that develops the best performance of the algorithms.

3. The 2 regression models with the best predictive performance were hyperparameter tuned to adapt them to the particular task and conditions on which they were chosen to work.
- Decision Tree hyperparameters tuned:
-** Random Forest hyperparameters tuned:**

4. A final algorithm was chosen to develop the regression machine learning algorithm.
- **Random Forest hyperparameters tuned.**

# Project Description

## Exploratory Data analysis

Most of the features in our dataset are in a categorical nominal format.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture1.JPG)

There are certain features in our dataset that have a considerably limited number of entries.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture2.JPG)

- Does price vary with the duration of the flight?

Our data shows that there is a correlation between the price and the duration of flights, where a ticket seems to be more expensive when the duration of the flight is shorter.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture3.JPG)

- How is the price affected by the days of booking?

Prices seem to be more expensive when the flight day is closer and drastically cheaper 20 days before.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture4.JPG)

# Data Normalization

Should we apply any special treatment for categorical data?

It was decided to use a label encoder approach instead of a hot encoder to avoid adding more columns to the data frame.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture5.JPG)

Being aware of the problems that changing the data to a numerical ordinal format could cause, and considering that our data follows a normal distribution, a min_max_scl approach was performed after.

# Data Cleaning

Should we delate any feature?

To analyse whether we are delating a column, it was decided to heatmap the correlation of features between them and the predictive feature, price.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture6.JPG)

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture7.JPG)

It was also decided to analyse the outliers and, after normalising the data, it turned out that the only column showing outliers was 'price_euro', which makes sense since almost all the other columns were categorical nominal values before normalisation and the outliers could correspond to the significantly lower number of business class tickets and the fact that they are significantly more expensive.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture8.JPG)

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture9.JPG)

Not further analysis have been decided to be done since we do want to keep those entries which correspond to business class tickets.

There seems to be a high correlation between duration and total stops with price, so we decide to keep these features long with departure time, arrival time, date of week and days remaining, as they may also be useful for further analysis.

Therefore, based on the analysis carried out, the following features were deleted: additional_info, route, date_of_journey, class, destination, source, and airline.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture10.JPG)

# Modelling Testing

Make any difference in the algorithm performance the way the data is train-testing split?

As a way to perform the modelling testing:

- 3 different train-testing split approaches were considered, in order to see which one performs better:
  - 80-20%
  - 90-10%
  - 70-30%
- 7 different algorithms were analysed:
  - Linear Regression
  - Ridge Regression
  - Lasso Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor
  - K-Nearest Neighbours Regressor

It turned out that the optimal models were Decision Tree Regressor and Random Forest Regressor, and the most effective train-testing split was 70% training and 30% testing.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture11.JPG)

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture12.JPG)

From the results we can see that:
- Decision tree:
  - Train score: It explain how the model generalized or fitted in the training data.
    - The model indicates that the algorithm performs exceptionally well on the training data, explaining approximately 99.5% of the variance.
  - Mean Squared Error: It explains the average squared difference between observed and predicted values.
    - The model highlights a reasonable score of 0.82, but there is still potential for improvement.
  - R-squared: It shows how well the model predicts the outcome of the dependent variable.
    - In the model we can see that 0.82 is a reasonable score, but there is still scope for enhancement.
  - Mean Absolute Error: It is the average variance between the significant values in the dataset and the projected values in it.
    - The analysis of the model indicates that, on average, its predictions deviate from the actual values by approximately 8 units.
- Random Forest:
  - Train score:
    - In the model, it is evident that the algorithm performs exceptionally well on the training data, accounting for approximately 92.9% of the variance.
  - Mean Squared Error:
    - Upon analysis, the model indicates that the Random Forest model's predictions are, on average, closer to the actual values than the Decision Tree, with a difference of 328.007.
  - R-squared:
    - The model shows a score of 0.88, indicating that the Random Forest model explains a higher percentage of the dependent variable's variance.
  - Mean Absolute Error
    - In the model we can see that 9.67 suggests that, on average, the Random Forest model's predictions have a slightly larger absolute difference from the actual values.

# Hyperparameter Tuning

After analysing the 7 different algorithms from which the study began, it was determined that only the Decision Tree and Random Forest displayed superior performance.

Random Forest was then chosen to undergo a hyperparameter tuning analysis based on its better overall performance in previous analyses and as a prime example of an algorithm unlikely to over or underfit.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture13.JPG)

After refining the hyperparameters, the Random Forest Regressor presented superior performance with an elevated R-squared (0.943) and decreased Mean Squared Error (236.418), showcasing improved predictive accuracy, and better fit to the test data relative to the prior configuration.

Upon analysing the model's performance, it is evident that it effectively predicts economic-class tickets and tickets with a low price. However, its performance is poor when dealing with business-class tickets or those priced higher than 500€. This can be attributed to the limited amount of business-class data, as previously discussed and proven.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture14.JPG)

# Results

As seen in the outcome, the model achieved a flight ticket price forecast with a mere £14.52 discrepancy between the projected and factual prices.

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture15.JPG)

![image](https://github.com/EduardoJMatosRomero/MachLearnCA1/blob/main/images/Capture16.JPG)





