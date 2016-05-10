# Boston Housing Price Prediction

<p align ="center">
<img src = "http://media.bizj.us/view/img/2122231/house-for-sale*750xx1200-675-0-63.jpg">
</p>

#### Project Description

- The Boston housing market is highly competitive, and the goal is to be one of the best real estate agents in the area. To compete in the real estate market, a few basic machine learning concepts are applied in order to assist a client with finding the best selling price for their home. The Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. The objective is to build an optimal model based on a statistical analysis with the tools available and then utilize this model to estimate the best selling price for the client's home. Additional information on the Boston Housing dataset can be found [`here`](https://archive.ics.uci.edu/ml/datasets/Housing). The source code developed for this project can be found in the ipython notebook: [`boston_housing.ipynb`](https://github.com/Tsmith5151/Boston-Housing-Price-Prediction-/blob/master/boston_housing.ipynb)

#### Software and Libraries
- Python 2.7
- NumPy
- scikit-learn
- iPython Notebook

### Statistical Analysis and Data Exploration
#### Using the NumPy library, calculate a few meaningful statistics about the dataset:
  
- The following code performs several statistical calculations on the dataset.

```python 
def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    # Size of data (number of houses)?
    number_houses = housing_features.shape[0]
    print "Number of houses:", number_houses
   
    # Number of features?
    number_features = housing_features.shape[1]
    print "Number of features:", number_features
   
    # Minimum price?
    min_price = np.min(housing_prices)
    print "Minimum Housing Price: ${:,.0f}".format(min_price*1000)

    # Maximum price?
    max_price = np.max(housing_prices)
    print "Maximum Housing Price: ${:,.0f}".format(max_price*1000)

    # Calculate mean price?
    mean_price = np.mean(housing_prices)
    print "Mean Housing Price: ${:,.0f}".format(mean_price*1000)

    # Calculate median price?
    median_price = np.median(housing_prices)
    print "Median Housing Price: ${:,.0f}".format(median_price*1000)

    # Calculate standard deviation?
    std_price = np.std(housing_prices)
    print "Standard Deviation: ${:,.0f}".format(std_price*1000) 
```  
|Statistics          | Value    |
| -------------      | ---------|
| Number of Houses   | 506      |
| Number of Features | 13       |
| Min Price          | $5,000   |
| Max Price          | $50,000  |
| Mean Price         | $22,533  |
| Median Price       | $21,200  |
| Standard Deviaton  | $9,188   |

#### Question 1: Of the available features for a given home, choose three you feel are significant and give a brief description for each of what they measure. 

- The scatter plots shown below consists of the median value of homes in Boston (`MEDV`) vs the 12 different features. Based on the results, `CRIM`, `RM`, and `LSAT` show a strong linear correlation for predicting the `MEDV`. For instance, as the number of rooms per house increases, so does the housing price. ![](MEDV.vs.Features.png)
 
#### Question 2: Using your client's feature set in the template code, which values correspond to the chosen features?
  - Applying LinearRegression, the following beta coefficients are listed in the table below, which can provide further insight into the features correlation with the housing price. 
```python
regression = LinearRegression()
regression.fit(X,y)
pd.DataFrame(zip(boston.feature_names, regression.coef_), columns = ['Features', 'Estimated Coef.'])
```
|Features | Coefficient |
| --------| ----------- |
| CRIM    | -0.1071     |
| ZN      | 0.0463      |
| INDUS   | 0.0208      |
| CHAS    | 2.6885      |
| NOX     | -17.7957    |
| RM      | 3.8047      |
| AGE     | 0.0007      |
| DIS     | -1.4757     |
| RAD     | 0.3046      |
| TAX     | 0.3056      |
| PTRATIO | -0.0123     |
| B       | 0.0093      |
| LSTAT   | -0.5254     |

### Evaluating Model Performance
#### Question 3: Why do we split the data into training and testing subsets? 

- Splitting the data into training and testing provides a measurement on how well the model will perform on out-of-sample data. Ideally, you want to perform a test on the dataset in which the algorithm has not seen in order to exclude any memorization. From there, you can then determine whether the algorithm works well with the out-of-sample data. A method of splitting the data using [`sklearn.cross_validation.train_test_split()`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.train_test_split.html) function allows the sample set to be randomly shuffled and then assigned `70%` as training and the remaining `30%` as the testing set. For this project, out of the `506` sampled houses, `354` are randomly selected and classified as the training set and the model is tested on `152` of the houses. 

 ``` python
 def split_data(city_data):
    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=RandomState)
    return X_train, y_train, X_test, y_test
```
#### Question 4:Which performance metric below is most appropriate for predicting housing prices and analyzing error? Why?

- Accuracy
- Precision
- Recall
- F1 score
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

- Being a regression problem and not classification, the metric used to evaluate the performance of the predicting variable (housing price) is the [`mean_squared_error`[(http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html), which takes all of the errors, the vertical distance from the observation to the regession line,squares  them, then finds the average. The MSE will heavily penalize outliers as oppose to the [`mean_absolute_error`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html),which takes all the errors, get the absolute values of them, and find the average.. When selecting betweeen the metrics, the objective is the minimize the error, which is why it is better to consider the MSE in the model. The two figures below dipicts the difference between the MSE (left) and the MAE (right) at a given depth `5`; it can be observed the error is minimized more for the MSE as opposed to the MAE. Furthermore, the machine learning algorithm for this project is [`sklearn.tree.DecisionTreeRegressor`](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html), therefore MSE is the valid peformance metric input for determining a split when constructing this model. 

```python
def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""
    #mae = mean_absolute_error(label, prediction)
    mse = mean_squared_error(label, prediction)
    return mse
    pass
```
 Mean Squarred Error             |  Mean Absolute Error  |
:-------------------------:|:-------------------------: 
![](learning_curve_5.png) | ![](learning_curve_mae_5.png) |

#### Question 5: What is the grid search algorithm and when is it applicable?

- A machine learning model can be fine-tuned by using the grid search algorithm, a hpyerparameter optimization technique. The performance of the model depends on the hyperparameters provided to the algorithm when training the model. It is desirable to find the right combination of the hyperparameters during training. [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV) is an exhaustive search that trains and evaluates a model for all possible combination of hyperparameters that produce the best model. The `1-10`), and a scoring function so it is able to evaluate the parameter that performed the best (performance `metric mean_squared_error`). 

```python
reg = GridSearchCV(regressor, parameters, scoring = mse_scoring)
```
#### Question 6: What is cross-validation and how is it performed on a model? Why would cross-validation be helpful when using grid search?
- A common occurance from machine learning algorithms is a degree of bias that can exist when random sampling the data and splitting the data between training and testing. To avoid sampling issues which can cause the training set to be too optimistic, cross-validation is a statistical approach that computes the average on multiple test sets. One of the most common iterators that performs the k-fold cross-validation is `KFolds` and the steps are as follows:
  - Splits the data into K equal folds
  - Uses one of the fold as the testing set and the remaining as the training set
  - Trains and records the test set results
  - The second and third steps are repeated using a different fold as the testing set each time
  - Calculate the average and standard devation of all the k-folds
  - Documentation can be found here:[`sklearn.cross_validation.KFolds()`](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.KFold.html)

- The `train_test_split()` method can be very simple to perform however the downside is the training data is reduced by 30% when finding the best regression method. Using the default `GridSearchCV` parameters, the `cross-validation` generator is set to 3-fold CV. So for instance, the housing dataset consists of `506` observations per feature and if k=3, `169` observations would be in each fold. The initial iteration entails the first fold containing the testing data (`169 observations`) and the remaining folds (2-3) containing `337` observations, `168` each. In second iteration, the testing set is now fold #2 while fold #1 and fold #3 are the training set. We train K different models, with each time leaving out a single subset for measuring the cross-validation error. The final cross-validation error is calculated by taking the mean or median of the K models. 
The advantage of cross-validation provides a more accurate estimate of the out-of-sample accuracy and is more efficient as every observation is used for both training and testing as oppose to the train/test/split method.  

## Analyzing Model Performance
#### Question 7: Choose one of the learning curve graphs your code creates. What is the max depth for the model? As the size of the training set increases, what happens to the training error? Describe what happens to the testing error.

- The learning curves for `max_depth` from 3-5 are shown below. As observed form the figures, the training error is virtually zero as the model has basically "memorized" the small training set. However, as more data is added to the model, the training error begins to increase. When the training size increases, error begins to increase and the high error translates to underfitting the data; likewise, the low training error would indicate overfitting the data. testing error is relatively high initially because it has not seen enough examples, but begins to decrease as more training examples are given. Based on the learning curve plots, it is concluded that the maximum depth for this model is `4`.

Max Depth = 3             |  Max Depth = 4  | Max Depth = 5 |
:-------------------------:|:-------------------------: | :-----------------------
![](learning_curve_3.png)  |  ![](learning_curve_4.png) | ![](learning_curve_5.png)

#### Question 8: Look at the learning curve graphs for the model with a max depth of 1 and a max depth of 10. When the model is using the full training set, does it suffer from high bias or high variance when the max depth is 1? What about when the max depth is 10?

- In the case of `max_depth` = `1`, the training error begins to significanlty increase as the training size increases and likewise, the error for the test error remains relatively high. Thus this situation where the `max_depth` of the decision tree is `1`, the model would be underfitting as the learning value is restricted to one level of the decision tree and does not allow the training set to learn data adequately. A low complexity decision tree results in high bias. For a `max_depth` = `10`, this would clearly illustrate a high variance and overfitting the data as shown below (right). The model has virtually memorized the training data but will not be expected to perform well with out-of-sample data.
 
Depth = 1                  | Depth = 10                 | 
:-------------------------:|:-------------------------: |
![](learning_curve_1.png)  |![](learning_curve_10.png)  | 

#### Question 9: From the model complexity graph, describe the training and testing errors as the max depth increases. Based on your interpretation of the graph, which max depth results in a model that best generalizes the dataset? Why?

- As the `max_depth` is increased, the training error begins to exponentially decline and approaches zero as the model becomes more complex. The testing error begins to differs from the training error near an `max_depth` = `4`, which best generalizes the dataset. At the instance where the model begins to behave too much like the training data then it might be observed that overfitting is occuring as the model does not perform well when testing and the error rates will begin to significanlty differ. A decrease in the training error implies that the model is becoming better at fitting the data; when the testing error plateaus and is no longer decreasing, additional knowledge is not gained on the out-of-sample data. If the error does not reduce any further during testing, then the complexity is increased for no reason and therefore overfitting occurs.
<p align="center">![](Model.Complexity.png)</p>

### Model Prediction
#### Question 10: Using grid search, what is the optimal max depth for your model? How does this result compare to your initial intuition?

```python
def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""
    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    # Setup a Decision Tree Regressor
    
    regressor = DecisionTreeRegressor()
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    mse_scoring = make_scorer(mean_squared_error, greater_is_better=False)
    
    #use grid search to fine tune the Decision Tree Regressor and
    #obtain the parameters that generate the best training performance. 

    reg = GridSearchCV(regressor, parameters, scoring = mse_scoring)
    reg.fit(X,y)
    
    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    print (reg.fit(X, y))    

    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    x = np.array(x)
    x = x.reshape(1, -1)
    y = reg.predict(x)

  print "House: " + str(x)
  print "Prediction: " + str(y)
  print "Best Parameters: ", reg.best_params_
  print "Best Estimator:", reg.best_estimator_
```
- Based on the model complexity graph, the model that best generalizes the data is when `max_depth` = `4`. Calling the `grid.best_params_` from `GridSearchCV` confirms the maximum level to split the decision tree is `4`. 

```python
#Optimization Results:
Best Parameters:  {'max_depth': 4}
Best Estimator: DecisionTreeRegressor(criterion='mse', max_depth=4, max_features=None, max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, presort=False, random_state=None, splitter='best')
```

#### Question 11: With your parameter-tuned model, what is the best selling price for your client?s home? How does this selling price compare to the statistics you calculated on the dataset? 

- Once the model has been trained and tested using the methods discussed above, we can now predict the housing price on the out-of-sample data provided by the client as shown in the table below. Statistical analysis performed on the dataset indicates the median home value is `$21,200` and the standard deviation is `$9,190`. Using the out-of-sample data, the model is predicting the average price of the home to be `$21,629`. Being the predicted value falls within one standard deviation of the mean, therefore the predication is considered to be reasonable. 

```python
    pd.DataFrame(zip(boston.feature_names, x), columns = ['Features', 'Client_Features'])
```
|Features | Sample      |
| --------| ----------- |
| CRIM    | 11.95       |
| ZN      | 0.00        |
| INDUS   | 18.10       |
| CHAS    | 0.00        |
| NOX     | 0.6590      |
| RM      | 5.6090      |
| AGE     | 90.00       |
| DIS     | 1.385       |
| RAD     | 24.0        |
| TAX     | 680.0       |
| PTRATIO | 20.20       |
| B       | 332.09      |
| LSTAT   | 12.13       |

#### Question 12: In a few sentences, discuss whether you would use this model or not to predict the selling price of future clients? homes in the Boston area.

- Being there is variability in the predicted price, which is actually expected since the code randomizes the data and therefore affects fitting the model to the data, multiple iterations of the model is desired in order to provide a conclusive prediction. Ideally, you would want to run the model multiple times and determine a range of values to that has the highest frequency of occurance and identify a central tendency based on the distribution for the `MEDV`. To do this, `1,000` iterations (randomly sampling each iteration, `cross-validation` =`3` ) were performed by using  `sklearn.GridSearchCV()` to determine the `max_depth` of the decision tree and the predicted housing price. Both results validate that the `max_depth` = `4` and the suggested value of the home is consistent around `$21,500`.  

Max_Depth                  | Predicted Prices           | 
:-------------------------:|:-------------------------: |
![](Gridsearch_MaxDepth_Hist.png)  |![](Gridsearch_Prices_Hist.png)  | 



#### Appendix:
   - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (=1 if tract bounds river;0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    % lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's
