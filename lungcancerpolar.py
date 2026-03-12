import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
df = pl.read_csv('dataset_med.csv')
print(df.head(5))
print(df.schema())
print("\n")
print(df.describe())
print("\n")
print(df.n_unique())
print(df.drop_nulls())
df.drop(columns=['id','diagnosis_date','end_treatment_date'], inplace=True)
print(df.head(5))
print(df.null_count())
print(df.drop_nulls())
df['survived'].value_counts()
print(df.corr(numeric_only=True))
sns.countplot(x='survived', data=df)
plt.show()
le = LabelEncoder()
for col in df.select_dtypes(include='object').shape[0]:
    df[col] = le.fit_transform(df[col])
#df['gender'] = le.fit_transform(df['gender'])

#print(df.select_dtypes(include='object').columns)
#print(df['survived'].value_counts())
X= df.drop(columns=['survived'])
y = df['survived']
'''print(X.shape)
X=pd.get_dummies(X,drop_first=True)
print(X.shape)
print(X.nunique().sort_values(ascending=False))'''

sm = SMOTE(random_state=42)
X_reasmpled , y_reasmpled = sm.fit_resample(X,y)
print(pl.Series(y_reasmpled).value_counts())
sns.countplot(x=y_reasmpled)
plt.show()
X_train,X_test,y_train,y_test = train_test_split(X_reasmpled,y_reasmpled,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#model1= LogisticRegression(random_state=42)
#model1.fit(X_train,y_train)
model = RandomForestClassifier(random_state=42, n_jobs=-1 ,n_estimators=100)
model.fit(X_train,y_train)
#model2 = XGBClassifier(random_state=42, n_jobs=-1, n_estimators=100)
#model2.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("RANDOM FOREST Accuracy:", accuracy_score(y_test, y_pred))
#print("Logestic Accuracy:", accuracy_score(y_test, model1.predict(X_test)))
#print("XGB Accuracy:", accuracy_score(y_test, model2.predict(X_test)))
print("Classification Report:\n", classification_report(y_test, y_pred))
#print("Logestic Classification Report:\n", classification_report(y_test, model1.predict(X_test)))
#print("XGB Classification Report:\n", classification_report(y_test, model2.predict(X_test))) 
params = {
    'n_estimators': [100,200],
    'max_depth':[10,20,None],
    'min_samples_split':[2,5]
    }
grid_search = GridSearchCV(RandomForestClassifier(random_state=42,n_jobs=-1),params,cv =3 ,n_jobs=-1,scoring='accuracy')
grid_search.fit(X_train,y_train)
print("best parameters:", grid_search.best_params_)
print("best score:", grid_search.best_score_)