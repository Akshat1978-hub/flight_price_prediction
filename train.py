import pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
df=pd.read_csv("Clean_Dataset.csv")
if "Unnamed: 0" in df.columns: df=df.drop(columns=["Unnamed: 0"])
X=df.drop(columns=["price"]); y=df["price"]
cat=X.select_dtypes(include="object").columns
num=[c for c in X.columns if c not in cat]
pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),cat)],remainder="passthrough")
model=Pipeline([("pre",pre),("model",RandomForestRegressor(n_estimators=100,random_state=42))])
model.fit(X,y)
joblib.dump(model,"model.pkl")
print("saved model.pkl")
