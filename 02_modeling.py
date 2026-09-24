"""Stage 2: modelling on Stage 1's CSV; no dataset reload or leakage."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
ROOT=Path(__file__).parent; OUT=ROOT/"output"
def prep():
    return ColumnTransformer([("num", Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]), ["pclass","age","sibsp","parch","fare"]), ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["sex","embarked"])])
def metrics(name, pipe, xtr, xte, ytr, yte):
    pipe.fit(xtr,ytr); p=pipe.predict(xte); score=pipe.predict_proba(xte)[:,1]
    return {"model":name,"accuracy":accuracy_score(yte,p),"precision":precision_score(yte,p),"recall":recall_score(yte,p),"f1":f1_score(yte,p),"auc":roc_auc_score(yte,score),"confusion_matrix":confusion_matrix(yte,p).tolist()}, pipe
if __name__=="__main__":
    OUT.mkdir(exist_ok=True); source=OUT/"titanic_cleaned.csv"
    if not source.exists(): raise FileNotFoundError("Run 01_eda.py first.")
    df=pd.read_csv(source); features=["pclass","age","sibsp","parch","fare","sex","embarked"]; X=df[features]; y=df.survived
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    models={"Logistic Regression":LogisticRegression(max_iter=1000),"Decision Tree":DecisionTreeClassifier(max_depth=4,random_state=42),"Random Forest":RandomForestClassifier(n_estimators=250,random_state=42)}
    rows=[]; fitted={}
    for name,est in models.items():
        row,pipe=metrics(name,Pipeline([("preprocess",prep()),("model",est)]),Xtr,Xte,ytr,yte); rows.append(row); fitted[name]=pipe
    comparison=pd.DataFrame(rows); comparison.to_csv(OUT/"classifier_comparison.csv",index=False)
    for name,pipe in fitted.items():
        RocCurveDisplay.from_predictions(yte,pipe.predict_proba(Xte)[:,1],name=name)
    plt.savefig(OUT/"roc_curves.png"); plt.close()
    names=fitted["Decision Tree"].named_steps["preprocess"].get_feature_names_out(); plt.figure(figsize=(18,8)); plot_tree(fitted["Decision Tree"].named_steps["model"],feature_names=names,class_names=["not survived","survived"],filled=True); plt.savefig(OUT/"decision_tree.png",bbox_inches="tight"); plt.close()
    variants={"baseline":LogisticRegression(max_iter=1000),"class_weight_balanced":LogisticRegression(max_iter=1000,class_weight="balanced"),"smote_train_only":LogisticRegression(max_iter=1000)}; imbalance=[]
    for label,est in variants.items():
        pipe=(ImbPipeline([("preprocess",prep()),("smote",SMOTE(random_state=42)),("model",est)]) if label=="smote_train_only" else Pipeline([("preprocess",prep()),("model",est)])); row,_=metrics(label,pipe,Xtr,Xte,ytr,yte); imbalance.append(row)
    pd.DataFrame(imbalance).to_csv(OUT/"imbalance_comparison.csv",index=False)
    grid=GridSearchCV(Pipeline([("preprocess",prep()),("model",RandomForestClassifier(oob_score=True,bootstrap=True,random_state=42))]),{"model__n_estimators":[100,250],"model__max_depth":[None,6,10],"model__max_features":["sqrt",.7]},cv=5,scoring="f1"); grid.fit(Xtr,ytr); oob=grid.best_estimator_.named_steps["model"].oob_score_
    reg_features=["pclass","age","sibsp","parch","sex","embarked"]; xr=df[reg_features]; yr=df.fare; a,b,c,d=train_test_split(xr,yr,test_size=.2,random_state=42)
    regprep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["pclass","age","sibsp","parch"]),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),["sex","embarked"])]); reg=Pipeline([("preprocess",regprep),("model",LinearRegression())]); reg.fit(a,c); pred=reg.predict(b); r2=r2_score(d,pred); adj=1-(1-r2)*(len(d)-1)/(len(d)-regprep.fit_transform(a).shape[1]-1)
    plt.scatter(pred,d-pred,alpha=.5); plt.axhline(0,color="red"); plt.xlabel("Predicted fare"); plt.ylabel("Residual"); plt.savefig(OUT/"regression_residuals.png"); plt.close()
    best_name=comparison.sort_values("f1",ascending=False).iloc[0].model; best=fitted[best_name]; joblib.dump(best,OUT/"best_pipeline.joblib"); assert len(joblib.load(OUT/"best_pipeline.joblib").predict(Xte.head(2)))==2
    report="\n\n# Modelling run report\n\nClass balance:\n"+y.value_counts(normalize=True).to_markdown()+"\n\n## Classifiers\n"+comparison.to_markdown(index=False)+"\n\n## Imbalance\n"+pd.DataFrame(imbalance).to_markdown(index=False)+f"\n\nGrid best params: {grid.best_params_}; OOB: {oob:.3f}.\n\nRegression: MAE={mean_absolute_error(d,pred):.3f}, RMSE={mean_squared_error(d,pred)**.5:.3f}, R2={r2:.3f}, Adjusted R2={adj:.3f}. Inspect residual plot for non-random spread (heteroscedasticity).\n\nRecommendation: deploy {best_name} initially because it has the highest held-out F1 ({comparison.set_index('model').loc[best_name,'f1']:.3f}) while retaining its pipeline preprocessing. Revisit the choice if recall or precision is operationally more important."
    with (OUT/"report.md").open("a",encoding="utf-8") as f:f.write(report)
