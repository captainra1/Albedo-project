# Albedo-project
## A Machine Learning Project.
studying lunar and mercury's albedo from the chemical data of planets collected by NASA by its space missions.

# About Project.
The Prospector mission launched by NASA has collected the data about various characteristics of the Lunar surface i.e,chemistry, mineralogy, and history of the surface and mapped them in various datasets. The data collected is not entirely independent and it is also incomplete.
Thus,A machine Learning model is to be modeled based on a regression model in order to identify the relation between the datasets collected by the prospector mission which can also make predictions about the chemical composition of the Lunar surface(incomplete dataset).

# Final goal of the Project
To develop a deep multi-objective machine learning model, which can make accurate predictions about the chemical composition of the Lunar surface.

# Outline of My Approach
The plan is based on the recurring model of software development. This project can be divided into three parts.
### Basic Data Preprocessing
- This is the basic but most important step of machine learning model development. This step will follow the given points.
To visualize the given datasets with different types of plots in order to develop a proper understanding of the datasets.
```Python
               import seaborn as sns
               import matplotlib.pyplot as plt
               sns.heatmap(‘dataset’)
               plt.imshow(‘dataset’) 
```
Our dataset will be in pixels, so we can visualize them in the image form either by using seaborn library’s function ‘heatmap()’ or matplotlib.pyplot’s ‘imshow()’.
```Python
               plt.hist(‘dataset’)
               
```
 We can use histogram plot in order to see the distributions of the pixels given in the datasets.
- Splitting the testing and training datasets if required.
- Applying basic feature engineering and other techniques (normalization and scaling) in order to make the dataset properly usable and well distributed so that the machine     learning model can give better output.

### Machine learning modeling
- After preparing the data, machine learning model is to be trained and made to predict desired values. In this step, given points are to be followed-
This project is based on a multi-output regression problem so regression algorithms are to be used. There are many regression algorithms like LinearRegression,       RandomForestRegressor, DecisionTreeRegressor and Ridge, etc which are used in regression problem normally but, this project is based on planetary dataset which is difficult to model and requires more precision so, complicated and having high computational cost algorithms are to be used like SupportVector Regressor(SVR) and Neural Network Regression.But, SVR doesn’t support multi-output so, to resolve this ‘multi-output Regression()’ can be used. Pseudo code for Neural network regression-
```python
             from keras.models import Sequential
             from keras.layers import Dense
             model=Sequential()
             model.add(Dense(m, input_dim=n_inputs,
             kernel_initializer='he_uniform', activation='relu')
             model.add(Dense(no of units in hidden layers))
             model.add(Dense(No. Of outputs))
             model.compile(loss='mae', optimizer='adam')
```

For support vector regression(SVR)-
             
```python
             from sklearn.multioutput import MultiOutputRegressor
             from sklearn.svm import LinearSVR
             model=LinearSVR()
             mult_obj_model=MultiOutputRegressor(model)
```
Normal algorithms can also give better results, but here main focus will be on Neural network regression and SVR.
- After fitting the model with the datasets,In order to validate the model, it’s accuracy/error is to be evaluated.For the accuracy score evaluation Cross-validation score will be used. model with high accuracy score will be selected.
- After selecting the best model from above, values will be predicted on the test set and accuracy will be measured by ‘r2_score()’ or ‘mean_squared_error()’ value.

### Debugging Cycle
This will be the last and the longest step.Here, deep understanding will be made on the best-selected model from the above step by plotting different plots i.e, residual plots, histogram plots, etc., and by checking bias and variance in the model. Based on the conclusion of the analysis, advanced feature engineering will be applied over the training dataset and the dataset will be again subjected to step-2(Machine learning modeling)  and this process will last until the desired accuracy is achieved.
- In case of Neural Network Regression based on the conclusion of ‘bias and variance’ in the model, to improve the accuracy, the hidden layers and no. of units will be changed.

## Road map of My approach
![image](https://github.com/captainra1/images/blob/master/rd_map.png)

## Lunar albedo
![lunar surface albedo image](https://github.com/captainra1/images/blob/master/lunar_al.png)

## Mercury upper surface albedo

![Mercury upper surface albedo image](https://github.com/captainra1/images/blob/master/mer_up_al.png)

## Mercury lower surface albedo
![Mercury lower surface albedo image](https://github.com/captainra1/images/blob/master/mer_low_al.png)

