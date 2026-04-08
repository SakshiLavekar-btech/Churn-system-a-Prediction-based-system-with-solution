import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,StackingClassifier
import xgboost as xg
from sklearn.metrics import confusion_matrix,classification_report
import pickle 

df=pd.read_csv('Telecom_churn.csv')

print(df.shape)
print(df.info())
print(df.describe())
print(df.sample(5))
print(df.isnull().sum())
print(df.duplicated().sum())
numerical_values=df.select_dtypes(include=['int','float'])
print(numerical_values.corr())

churn_by_gender = df.groupby("SeniorCitizen")["Churn"].value_counts(normalize=True)
# customerID,gender will be excluded during training
# customerid has no null values eg 7296-PIXQY dont add in training set
# gender has no null value uniue values ['Female' 'Male']
# SeniorCitizen no null values [0 1]
# Partner 0 ['Yes' 'No']
# Dependents 0['No' 'Yes']
df['Partner']=df['Partner'].map({'No':0,'Yes':1})
df['Dependents']=df['Dependents'].map({'No':0,'Yes':1})
df['family']=df['Partner']+df['Dependents']+1
print(df['family'])
# tenure 0 65,19 etc
churn_by_tenure=df.groupby('tenure')['Churn'].value_counts(normalize=True)
print(churn_by_tenure)
# PhoneService o null Yes    6361 No      682
# MultipleLines 0 No                  3390 Yes                 2971 No phone service     682
df['MultipleLines']=df['MultipleLines'].map({'No phone service':0,'No':1,'Yes':2})

# InternetService null values 0  Fiber optic    3096 DSL            2421 No             1526
# OnlineSecurity 0  No                     3498 Yes                    2019 No internet service    1526
# DeviceProtection 0  No                     3095 Yes                    2422 No internet service    1526
# TechSupport 0  No                     3473 Yes                    2044 No internet service    1526
# StreamingTV 0  No                     2810 Yes                    2707 No internet service    1526
# StreamingMovies 0 No                     2785 Yes                    2732 No internet service    1526
#crosstab tells relation between two enties
a=pd.crosstab(df["StreamingTV"], df["Churn"], normalize="index")
b=pd.crosstab(df["StreamingMovies"], df["Churn"], normalize="index")
print(a,b)
df['StreamingMovies']=df['StreamingMovies'].map({'No internet service':0,'No':0,'Yes':1})
df['StreamingTV']=df['StreamingTV'].map({'No internet service':0,'No':0,'Yes':1})
df['Entertainment']=df['StreamingMovies']|df['StreamingTV']

df['OnlineSecurity']=df['OnlineSecurity'].map({'No internet service':0,'No':0,'Yes':1,})
df['DeviceProtection']=df['DeviceProtection'].map({'No internet service':0,'No':0,'Yes':1,})
df['TechSupport']=df['TechSupport'].map({'No internet service':0,'No':0,'Yes':1,})
df['PaperlessBilling']=df['PaperlessBilling'].map({'No':0,'Yes':1,})
df['OnlineBackup']=df['OnlineBackup'].map({'No internet service':0,'No':0,'Yes':1,})

# 0 Contract Month-to-month    3875 Two year          1695 One year          1473
# 0 PaperlessBilling Yes    4171 No     2872
# 0 PaymentMethod Electronic check             2365 Mailed check                 1612 Bank transfer (automatic)    1544 Credit card (automatic)      1522
# 0 MonthlyCharges nice spread 20 max
# 0 TotalCharges 20.2  
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df[["tenure","MonthlyCharges","TotalCharges"]].corr())
# churn
# print(df.shape)
# print(df['Churn'])
df['Churn']=df['Churn'].map({'No':0,'Yes':1})
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df = df[['customerID', 'gender', 'SeniorCitizen', 'family',
         'tenure', 'MultipleLines', 'InternetService',
         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
         'Entertainment', 'Contract', 'PaperlessBilling',
         'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']]

X=df.iloc[:,2:-1]
y=df.iloc[:,-1]
print(X.info())
print(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
print(X_train.shape)

nominal_columns=['InternetService','PaymentMethod']
ordinal_columns=['Contract']
scale=['tenure','MonthlyCharges','TotalCharges']
ohe=OneHotEncoder(drop='first')
binary_columns = ['SeniorCitizen', 'family',
          'MultipleLines',
         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
         'Entertainment', 'PaperlessBilling']
preprocessing=ColumnTransformer(
    transformers=[
    ('trf1',SimpleImputer(strategy='mean'),['TotalCharges']),
    ('trf2',OneHotEncoder(drop='first'),nominal_columns),
    ('trf3',OrdinalEncoder(categories=[['Month-to-month','One year','Two year']]),ordinal_columns),
    ('trf4',StandardScaler(),scale),
    ('bin', 'passthrough', binary_columns)
],remainder='drop')


pipe=Pipeline(
    [
        ('preprocessing',preprocessing),
        # ('model',LogisticRegression(random_state=42)),#Hyperparameters max_iter=1000,class_weight='balanced'
        # ('model',BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42),n_estimators=100))
        # ('model',xg.XGBClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42,use_label_encoder=False, eval_metric='logloss'))
 ('model', StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('bag', BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=100
            )),
            ('xg', xg.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ],
        final_estimator=LogisticRegression(),
        cv=5
#         0.8184397163120567
# [[468  43]
#  [ 85 109]]
#               precision    recall  f1-score   support

#            0       0.85      0.92      0.88       511
#            1       0.72      0.56      0.63       194

#     accuracy                           0.82       705
#    macro avg       0.78      0.74      0.75       705
# weighted avg       0.81      0.82      0.81       705
    ))
])

pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
print(y_pred)
print(accuracy_score(y_test,y_pred)) 
#0.0.8170212765957446 logistic regression

print(confusion_matrix(y_test,y_pred))
#logistic [[480  44][ 82  99]]
print(classification_report(y_test,y_pred))
# Logistic regression
#  precision    recall  f1-score   support

#            0       0.83      0.88      0.86       526
#            1       0.58      0.47      0.52       179

#     accuracy                           0.78       705
#    macro avg       0.70      0.68      0.69       705
# weighted avg       0.77      0.78      0.77       705

#DecisionTree with bagging
# 0.7829787234042553
# [[461  50]
#  [103  91]]
#               precision    recall  f1-score   support

#            0       0.82      0.90      0.86       511
#            1       0.65      0.47      0.54       194

#     accuracy                           0.78       705
#    macro avg       0.73      0.69      0.70       705
# weighted avg       0.77      0.78      0.77       705

#XGD Regressor
# 0.8127659574468085
# [[467  44]
#  [ 88 106]]
#               precision    recall  f1-score   support

#            0       0.84      0.91      0.88       511
#            1       0.71      0.55      0.62       194

#     accuracy                           0.81       705
#    macro avg       0.77      0.73      0.75       705
# weighted avg       0.80      0.81      0.80       705

pickle.dump(pipe,open('churn_system.pkl','wb'))