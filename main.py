import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from sklearn import tree
import graphviz


if __name__ == '__main__':
    # read input data and check which columns has missing values
    existing_customers = pd.read_csv("data/existing-customers.csv", sep=';', index_col=0)
    potential_customers = pd.read_csv("data/potential-customers.csv", sep=';', index_col=0)

    original_data = existing_customers.replace('', pd.NaT)
    original_data2 = potential_customers.replace('', pd.NaT)

    # fill in the missing values using most frequent using the SimpleImputer
    imputer = SimpleImputer(strategy="most_frequent")
    existing_customers = pd.DataFrame(imputer.fit_transform(existing_customers), columns=existing_customers.columns)
    potential_customers = pd.DataFrame(imputer.fit_transform(potential_customers), columns=potential_customers.columns)

    # convert categorical attributes to numerical attributes using the LabelEncoder
    le = LabelEncoder()
    categoricalColumns = ['class','workclass', 'education', 'marital-status', 'relationship','occupation', 'race', 'sex', 'native-country']
    categoricalColumnsPot = ['workclass', 'education', 'marital-status', 'relationship','occupation', 'race', 'sex', 'native-country']

    for col in categoricalColumns:
        existing_customers[col] = le.fit_transform(existing_customers[col])

    for col in categoricalColumnsPot:
        potential_customers[col] = le.fit_transform(potential_customers[col])

    # Normalize data of existing_customers using the MinMaxScaler
    scaler = preprocessing.MinMaxScaler()
    columnsToBeNormalized =  ['age', 'workclass', 'education', 'education-num', 'occupation', 'race', 'sex','capital-gain','capital-loss', 'hours-per-week', 'native-country']
    dataToBeNormalized = existing_customers[columnsToBeNormalized]
    dataToBeNormalizedPotential = potential_customers[columnsToBeNormalized]
    scaler.fit(dataToBeNormalized)
    scaler.fit(dataToBeNormalizedPotential)
    normalizedData = scaler.transform(dataToBeNormalized)
    normalizedDataPotential = scaler.transform(dataToBeNormalizedPotential)

    existing_customers[columnsToBeNormalized] = normalizedData
    potential_customers[columnsToBeNormalized] = normalizedDataPotential

    #splitting the data in a train and test set using the train_test_split function
    features = existing_customers.iloc[:, :-1]
    label = existing_customers.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3,random_state=42)

    #apply classifier1: GaussianNB()
    gnbModel = GaussianNB()
    gnbModel.fit(X_train, y_train)
    y_pred = gnbModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("GaussianNB")
    print(f"Accuracy: {accuracy}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    # apply classifier2: KNeighborsClassifier(n_neighbors=5)
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3,random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("KNeighborsClassifier")
    print("Accuracy:", accuracy)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    # try which k is the best
    kValues = range(1, 20)
    crossValisationResults = []
    for k in kValues:
        # Train a KNN classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)
        # Use cross-validation to evaluate the performance of the classifier
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3, random_state=42)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        crossValisationResults.append(np.mean(scores))
    plt.plot(kValues, crossValisationResults)
    plt.xlabel('Number of neighbors')
    plt.ylabel('Cross-validation score')
    plt.show()

    best_k = kValues[np.argmax(crossValisationResults)]
    print('Best k:', best_k)

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3,random_state=42)
    knn = KNeighborsClassifier(n_neighbors=17)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("KNeighborsClassifier 17")
    print("Accuracy:", accuracy)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


    #Decision tree
    clf = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best max depth:", grid_search.best_params_['max_depth'])
    print("Best score:", grid_search.best_score_)
    # Decision tree using best max depth 8
    clf = DecisionTreeClassifier(max_depth=8)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print("Accuracy:", accuracy)
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    featureNamse =  ['age', 'workclass', 'education', 'education-num', 'marital-status','occupation', 'relationship','race', 'sex','capital-gain','capital-loss', 'hours-per-week', 'native-country']
    #dot_data = export_graphviz(clf, out_file=None,
    #                           feature_names=featureNamse,
    #                           class_names="class",
    #                           filled=True, rounded=True,
    #                           special_characters=True)

    # Use the GraphViz library to display the decision tree
    #graph = graphviz.Source(dot_data)
    #graph.render("decisionTree.dot")  # save as pdf
    #graph.view()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #use bagging
    base_estimator = DecisionTreeClassifier()
    param_grid = {'n_estimators': [10, 50, 100, 150, 200]}
    bagging = BaggingClassifier(estimator=base_estimator)
    grid_search = GridSearchCV(bagging, param_grid=param_grid, cv=5)

    grid_search.fit(X_train, y_train)
    print("Bagging")
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
    #accuracy = bagging.score(X_test, y_test)
    #print("Accuracy:", accuracy)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    #use Boosting
    base_estimator = DecisionTreeClassifier(max_depth=8)
    ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=50, random_state=42)
    ada_boost.fit(X_train, y_train)

    y_pred = ada_boost.predict(X_test)
    accuracy = ada_boost.score(X_test, y_test)
    print("Boosting")
    print(f"Accuracy: {accuracy}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))


    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250]
    }
    gb_clf = GradientBoostingClassifier()
    grid_search = GridSearchCV(gb_clf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters with GridSearchCV :", grid_search.best_params_)
    print("Best Accuracy Score with GridSearchCV: ", grid_search.best_score_)
    print("GradientBoostingClassifier")
    gbt = GradientBoostingClassifier()
    gbt.fit(X_train, y_train)

    y_pred = gbt.predict(X_test)
    accuracy = gbt.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))


    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250]
    }
    # use classifier RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    accuracy = rfc.score(X_test, y_test)
    print("RandomForestClassifier")
    print(f"Accuracy: {accuracy}")
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))

    #check which estimator gives the best results
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150, 200, 250]
    }
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Accuracy Score:", grid_search.best_score_)


    # Predict the labels of potential customers
    X_pred = potential_customers.iloc[:, :]
    X = existing_customers.iloc[:, :-1]
    Y = existing_customers.iloc[:, -1]

    gbt = GradientBoostingClassifier()
    gbt.fit(X, Y)
    y_pred = gbt.predict(X_pred)

    #base_estimator = DecisionTreeClassifier(max_depth=8)
    #ada_boost = AdaBoostClassifier(estimator=base_estimator, n_estimators=250, random_state=42)
    #ada_boost.fit(X, Y)
    #y_pred = ada_boost.predict(X_pred)

    potential_customerss = pd.read_csv("data/potential-customers.csv", sep=';', index_col=0)
    potential_customers['class'] = y_pred
    potential_customers.to_csv("data/potential-customers-output.csv", index=False)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    #compute the distance between potential and existing clients
    # Take the top 10% potential high income clients with the smallest distance to the nearest existing client
    # The same of low income but instead of 10% we take 5%
    # Write top 10% and 5% rowIDs to corresponding file
    highIncomesExisting = existing_customers[existing_customers['class'] == 1]
    lowIncomesExisting = existing_customers[existing_customers['class'] == 0]
    highIncomesPotential = potential_customers[potential_customers['class'] == 1]
    lowIncomesPotential = potential_customers[potential_customers['class'] == 0]
    NumberHigh = round(len(highIncomesPotential) * 0.1)
    NumberLow = round(len(lowIncomesPotential) * 0.05)
    print("len(highIncomesPotential)",len(highIncomesPotential))
    print("len(lowIncomesPotential)",len(lowIncomesPotential))
    Profit = NumberHigh * 980 + NumberLow * (-310)
    print("Profit",Profit )
    TotalCost = (NumberHigh + NumberLow)*10
    print("TotalCost",TotalCost)
    TotalRevenue = Profit - TotalCost
    print("TotalRevenue",TotalRevenue)


    # For high income
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(highIncomesExisting)
    distancesHigh, indicesHigh = nbrs.kneighbors(highIncomesPotential)
    sorted_indices = np.argsort(distancesHigh[:, 0])
    selected_rowsHigh = sorted_indices[:291]
    selected_dataHigh = highIncomesPotential.iloc[selected_rowsHigh, :]
    selected_dataHigh.index.to_series().to_csv('PotentialNewHighIncomeClients.txt', index=False)

    # For low income
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(lowIncomesExisting)
    distancesLow, indicesLow = nbrs.kneighbors(lowIncomesPotential)
    sorted_indices = np.argsort(distancesLow[:, 0])
    selected_rowsLow = sorted_indices[:669]
    selected_dataLow = lowIncomesPotential.iloc[selected_rowsLow, :]
    selected_dataLow.index.to_series().to_csv('PotentialNewLowIncomeClients.txt', index=False)