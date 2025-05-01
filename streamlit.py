import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", \
                       "streamlit", "pandas", "numpy", "matplotlib", "seaborn", \
                       "scikit-learn"])

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV, cross_val_score,
                                     permutation_test_score)
from sklearn.metrics import accuracy_score, f1_score

# --- Helper functions ---
def load_dataset(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None


def merge_with_priority(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str="left") -> pd.DataFrame:
    df = left.merge(right, on=on, how=how, suffixes=("_left","_right"))
    overlap = {c[:-5] for c in df if c.endswith("_left")}.intersection({c[:-6] for c in df if c.endswith("_right")})
    for base in overlap:
        l, r = base + "_left", base + "_right"
        df[base] = df[l].combine_first(df[r])
        df.drop([l,r], axis=1, inplace=True)
    return df


def clean_and_impute(df, target_col=None, missing_thresh=0.2):
    # replace sentinel
    df = df.replace(-999, np.nan)
    # drop high-missing
    miss_frac = df.isna().mean()
    drop_cols = miss_frac[miss_frac>missing_thresh].index.tolist()
    df.drop(columns=drop_cols, inplace=True)
    # genus-median impute then global median
    if 'Genus' in df.columns:
        num = df.select_dtypes(include='number')
        filled = (pd.concat([df['Genus'], num], axis=1)
                  .groupby('Genus').transform(lambda x: x.fillna(x.median())))
        df[num.columns] = filled.fillna(df[num.columns].median())
    else:
        df = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(df), columns=df.columns)
    return df


def reverse_engineer_rules(df, diet_vars, combined_vars, target_col):
    df2 = df.dropna(subset=[target_col])
    X = df2[diet_vars+combined_vars]
    y = df2[target_col]
    dt = DecisionTreeClassifier(max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, random_state=0)
    dt.fit(X,y)
    rules = export_text(dt, feature_names=diet_vars+combined_vars)
    acc = f1_score(y, dt.predict(X), average='macro')
    return dt, rules, acc

# --- Streamlit UI ---
st.title("Mammalian Dietary Traits: DR & Decision-Tree")

st.sidebar.header("Data Inputs & Settings")
f1 = st.sidebar.slider("Nested CV folds (outer)", 2, 10, 5)
missing_thresh = st.sidebar.slider("Missing-data threshold", 0.0, 1.0, 0.2)

# File uploads
uploaded = {}
sources = ["reported","imputed","EltonTraits","MammalDiet2015","MammalDiet2018","AmnioteDb","DietGuild","Validation"]
for src in sources:
    uploaded[src] = st.sidebar.file_uploader(f"Upload {src} CSV", type="csv")

if all(uploaded.values()):
    # load all
    df_rep = load_dataset(uploaded['reported'])
    df_imp = load_dataset(uploaded['imputed'])
    elton = load_dataset(uploaded['EltonTraits'])
    md15  = load_dataset(uploaded['MammalDiet2015'])
    md18  = load_dataset(uploaded['MammalDiet2018'], skiprows=2)
    amio  = load_dataset(uploaded['AmnioteDb'])
    guild = load_dataset(uploaded['DietGuild'])
    val   = load_dataset(uploaded['Validation'])

    # rename and merge chain
    elton = elton.rename(columns={"Scientific":"scientific"})
    df_rep.rename(columns={"iucn2020_binomial":"scientific"}, inplace=True)
    df_imp = df_imp
    md15['scientific'] = md15['Genus']+" "+md15['Species']
    md18.rename(columns={"Binomial":"scientific"}, inplace=True)
    amio['scientific'] = amio['genus']+" "+amio['species']
    val.rename(columns={"ScientificNameFull":"scientific"}, inplace=True)
    # merge
    M = merge_with_priority(elton, md15, on='scientific', how='outer')
    M = merge_with_priority(M, md18, on='scientific')
    for d in [df_rep, df_imp, amio, guild, val]:
        M = merge_with_priority(M,d,on='scientific')
    st.subheader("Master dataset preview")
    st.dataframe(M.head())

    # clean & impute
    df_clean = clean_and_impute(M, missing_thresh)
    st.write(f"Data shape after clean & impute: {df_clean.shape}")

    # build combined diet vars
    diet_cols = [c for c in df_clean.columns if c.startswith('Diet.')]
    combined = [
        'Diet.VertTerrestrial','Diet.VertAll','Diet.AnimalsAll',
        'Diet.PlantHighSugar','Diet.PlantLowSugar','Diet.PlantAll'
    ]
    df_clean['Diet.VertTerrestrial'] = df_clean['Diet.Vend']+df_clean['Diet.Vect']+df_clean['Diet.Vunk']
    df_clean['Diet.VertAll'] = df_clean['Diet.VertTerrestrial'] + df_clean['Diet.Vfish']
    df_clean['Diet.AnimalsAll'] = df_clean['Diet.VertAll'] + df_clean['Diet.Inv']
    df_clean['Diet.PlantHighSugar'] = df_clean['Diet.Fruit']+df_clean['Diet.Nect']
    df_clean['Diet.PlantLowSugar'] = df_clean['Diet.Seed']+df_clean['Diet.PlantO']
    df_clean['Diet.PlantAll'] = df_clean['Diet.PlantHighSugar']+df_clean['Diet.PlantLowSugar']

    # target
    if 'DerekDietClassification90InsVertivoreSorting' in df_clean:
        df_clean['Diet_Class'] = df_clean['DerekDietClassification90InsVertivoreSorting']
    y = df_clean['Diet_Class']
    X = df_clean.drop(columns=['scientific','Diet_Class']+combined)

    # scale
    X_scaled = StandardScaler().fit_transform(X.select_dtypes(include='number'))
    y_enc = LabelEncoder().fit_transform(y)
    X_train,X_test,y_train,y_test = train_test_split(
        X_scaled,y_enc,test_size=0.2,stratify=y_enc,random_state=0)

    # reverse-engineer validation
    dt0, rules, acc0 = reverse_engineer_rules(val, diet_cols, combined, 'DerekDietClassification90InsVertivoreSorting')
    st.subheader("Overfit DT rules on validation set (perfect-fit)")
    st.code(rules)
    st.write(f"Baseline perfect-fit accuracy: {acc0:.3f}")

    # nested CV tuning
    st.subheader("Nested CV: DecisionTree + manual PCA")
    k_opt = st.sidebar.selectbox("# PCs for tuning", [5,10,15,20], index=1)
    pipe = Pipeline([('scale', StandardScaler()),
                     ('pca', FunctionTransformer(lambda X, k: (X-X.mean(axis=0)).dot(
                         PCA(n_components=k).fit(X).components_.T), kw_args={'k':k_opt})),
                     ('clf', DecisionTreeClassifier(random_state=0))])
    param = {'pca__kw_args':[{'k':k_opt}],
             'clf__max_depth':[None,5,10,20],
             'clf__min_samples_leaf':[1,5,10]}
    inner = StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
    outer = StratifiedKFold(n_splits=3,shuffle=True,random_state=1)
    grid = GridSearchCV(pipe,param,cv=inner,scoring='f1_macro',n_jobs=-1)
    scores = cross_val_score(grid, X_scaled, y_enc, cv=outer, scoring='f1_macro', n_jobs=-1)
    st.write("Nested CV F1_macro:", np.round(scores,3), "mean=", round(scores.mean(),3))

    # compare DR methods
    st.subheader("Direct comparison: Full vs. VT vs. PCA (precomputed)")
    # full
    dt_full = DecisionTreeClassifier(random_state=0).fit(X_train,y_train)
    f_full = f1_score(y_test, dt_full.predict(X_test), average='macro')
    # vt
    vt = VarianceThreshold(0.01).fit(X_train)
    Xv_test = vt.transform(X_test)
    f_vt = f1_score(y_test, DecisionTreeClassifier(random_state=0).fit(vt.transform(X_train),y_train).predict(Xv_test), average='macro')
    # pca
    Xp = PCA(n_components=k_opt).fit_transform(X_scaled)
    f_pca = f1_score(y_enc, DecisionTreeClassifier(random_state=0).fit(Xp,y_enc).predict(Xp), average='macro')
    df_comp = pd.DataFrame({
        'Method':['Full','VarianceThreshold',f'PCA k={k_opt}'],
        'F1_macro':[f_full,f_vt,f_pca]
    })
    st.table(df_comp)

    # visualization
    st.subheader("PCA Biplot")
    pca0 = PCA(n_components=2).fit(X_scaled)
    proj = pca0.transform(X_scaled)
    fig, ax = plt.subplots()
    sc = ax.scatter(proj[:,0],proj[:,1],c=y_enc,cmap='tab10',alpha=0.7)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    st.pyplot(fig)

else:
    st.info("Upload all required CSVs to begin.")
