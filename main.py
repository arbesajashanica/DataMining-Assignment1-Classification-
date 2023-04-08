import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # read input data and check which columns has missing values
    existing_customers = pd.read_csv("data/existing-customers.csv", sep=';')
    potential_customers = pd.read_csv("data/potential-customers.csv", sep=';')

    original_data = existing_customers.replace('', pd.NaT)
    original_data2 = potential_customers.replace('', pd.NaT)

    print(existing_customers.isna().sum())
    print(potential_customers.isna().sum())

    # fill in the missing values using most frequent using the SimpleImputer
    imputer = SimpleImputer(strategy="most_frequent")
    existing_customers = pd.DataFrame(imputer.fit_transform(existing_customers), columns=existing_customers.columns)
    potential_customers = pd.DataFrame(imputer.fit_transform(potential_customers), columns=potential_customers.columns)

    # convert categorical attributes to numerical attributes using the LabelEncoder
    le = LabelEncoder()
    categoricalColumns = ['workclass', 'education', 'marital-status', 'relationship','occupation', 'race', 'sex', 'native-country']
    for col in categoricalColumns:
        existing_customers[col] = le.fit_transform(existing_customers[col])

    print(existing_customers.iloc[19])
    print(existing_customers)

    # Normalize data of existing_customers using the MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    columnsToBeNormalized =  ['age', 'workclass', 'education', 'education-num', 'occupation', 'race', 'sex','capital-gain','capital-loss', 'hours-per-week', 'native-country']
    dataToBeNormalized = existing_customers[columnsToBeNormalized]
    scaler.fit(dataToBeNormalized)
    normalizedData = scaler.transform(dataToBeNormalized)
    existing_customers[columnsToBeNormalized] = normalizedData
    print(existing_customers.iloc[19])

    #splitting the data in a train and test set using the train_test_split function







