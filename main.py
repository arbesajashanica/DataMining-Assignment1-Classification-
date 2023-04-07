import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    existing_customers = pd.read_csv("data/existing-customers.csv", sep=';')
    potential_customers = pd.read_csv("data/potential-customers.csv", sep=';')

    original_data = existing_customers.replace('', pd.NaT)
    original_data2 = potential_customers.replace('', pd.NaT)

    print(existing_customers.isna().sum())
    print(potential_customers.isna().sum())

    # fill in the missing values using most frequent
    imputer = SimpleImputer(strategy="most_frequent")
    existing_customers = pd.DataFrame(imputer.fit_transform(existing_customers), columns=existing_customers.columns)
    potential_customers = pd.DataFrame(imputer.fit_transform(potential_customers), columns=potential_customers.columns)

    # convert categorical attributes to numerical attributes
    le = LabelEncoder()
    categoricalColumns = ['workclass', 'education', 'occupation', 'race', 'sex', 'native-country']
    for col in categoricalColumns:
        existing_customers[col] = le.fit_transform(existing_customers[col])

    print(existing_customers.iloc[19])


