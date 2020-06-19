import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import  style
import pickle

## NOTE : This is an example of G3 prediction with pd.DataFrames

# load the data from the csv file.Since our data is seperated by semicolons we need to do sep=";"
data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# data with no NaN values
data_no_nulls = data.dropna()

# X is our features we use to try and do our prediction
X = data_no_nulls.loc[:,["G1", "G2", "studytime", "failures", "absences"]]

# y is the value we try to predict
y = data_no_nulls.loc[:,["G3"]]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE

best=0
for x in range(30):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    regressor = LinearRegression()
    # train the model using the training data
    regressor.fit(x_train, y_train)
    acc = regressor.score(x_test, y_test)
    predictions = regressor.predict(x_test)
    #print("accuracy: \n", acc)
    #print("r^2 aka accuracy: \n", metrics.r2_score(y_test, predictions))
    if acc > best:
        best=acc
        # save our model with pickle
        # you save model if u have hundreds of thousand data you dont want to retrain the model every time
        # so you save that model if it has a good accuracy
        with open("student_model.pickle", 'wb') as f:
            pickle.dump(regressor, f)

#you load your saved model
pickle_in = open("student_model.pickle","rb")
regressor=pickle.load(pickle_in)

predictions_df = pd.DataFrame(predictions)
predictions_df.columns=['Predicted G3']
# Reset the index values to the second dataframe appends properly
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)

#mean squared error
print("mse:\n ",metrics.mean_squared_error(y_test,predictions))
# root mean squared error
print("rmse: \n ",np.sqrt(metrics.mean_squared_error(y_test,predictions)))

print('Coefficient: \n', regressor.coef_)
print('Intercept: \n', regressor.intercept_)


#Concat the two dataframes
merged_df = pd.concat([predictions_df, y_test_df],axis=1)
#show all rows
pd.set_option('display.max_rows', None)
print(merged_df)

# Drawing and plotting model
plot = "G2" # Change this to G1, G2, studytime or absences to see other graphs
plt.scatter(data_no_nulls[plot], data_no_nulls["G3"],alpha=0.3)
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
