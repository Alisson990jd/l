import sys, json, numpy as np, pandas as pd, xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
g=pd.read_parquet(sys.argv[1])
g["over25"]=((g.hg+g.ag)>2).astype(int); g["btts"]=((g.hg>0)&(g.ag>0)).astype(int)
team=[c for c in g.columns if c.startswith("elo_") or any(c.startswith(p) for p in ("gf_l","ga_l","pts_l","rest_days","n_prev"))]
lineup=[c for c in g.columns if c.startswith(("xi_","gk_","bench_"))]
FEATS={"times (Elo+forma)":team+["comp_type"],"times+escalação":team+lineup+["comp_type"]}
def rps(p,y):
    o=np.eye(3)[y]; return np.mean(np.sum((np.cumsum(p,1)-np.cumsum(o,1))[:,:2]**2,1)/2)
res=[]; preds=[]; imp=None
for test in (2022,2023,2024,2025):
    tr=g[g.season<test]; te=g[g.season==test]; va=tr[tr.season==test-1]; tr2=tr[tr.season<test-1]
    for target,k in (("result",3),("over25",2),("btts",2)):
        y_tr,y_te=tr[target].values,te[target].values
        prior=np.bincount(y_tr,minlength=k)/len(y_tr); P={"base (frequência)":np.tile(prior,(len(te),1))}
        lr=LogisticRegression(max_iter=1000).fit(tr[["elo_diff"]].fillna(0),y_tr); P["só Elo (logística)"]=lr.predict_proba(te[["elo_diff"]].fillna(0))
        for name,fs in FEATS.items():
            params=dict(n_estimators=2000,learning_rate=0.03,max_depth=4,subsample=0.8,colsample_bytree=0.7,min_child_weight=20,
                        objective="multi:softprob" if k==3 else "binary:logistic",eval_metric="mlogloss" if k==3 else "logloss",early_stopping_rounds=100,n_jobs=4)
            m=xgb.XGBClassifier(**params).fit(tr2[fs],tr2[target],eval_set=[(va[fs],va[target])],verbose=False)
            m2=xgb.XGBClassifier(**{**params,"n_estimators":m.best_iteration+1,"early_stopping_rounds":None}).fit(tr[fs],y_tr)
            p=m2.predict_proba(te[fs]); P[f"XGBoost {name}"]=p
            if target=="result" and name=="times+escalação" and test==2025:
                imp=pd.Series(m2.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        for mname,p in P.items():
            r={"test_season":test,"target":target,"model":mname,"n":len(te),"acc":accuracy_score(y_te,p.argmax(1)),"logloss":log_loss(y_te,p,labels=list(range(k)))}
            r["rps" if k==3 else "auc"]=rps(p,y_te) if k==3 else roc_auc_score(y_te,p[:,1])
            res.append(r)
        if target=="result":
            preds.append(te[["game_id","date","competition_id","home_club_name","away_club_name","hg","ag","result"]].assign(pH=P["XGBoost times+escalação"][:,0],pD=P["XGBoost times+escalação"][:,1],pA=P["XGBoost times+escalação"][:,2],
                pH_team=P["XGBoost times (Elo+forma)"][:,0],pD_team=P["XGBoost times (Elo+forma)"][:,1],pA_team=P["XGBoost times (Elo+forma)"][:,2]))
    print("done",test,flush=True)
R=pd.DataFrame(res); R.to_csv(sys.argv[2]+"/results.csv",index=False)
pd.concat(preds).to_parquet(sys.argv[2]+"/preds.parquet",index=False); imp.to_csv(sys.argv[2]+"/importance.csv")
pd.set_option("display.width",200)
print(R.groupby(["target","model"])[["acc","logloss","rps","auc"]].mean().round(4))
print(imp.head(20))
