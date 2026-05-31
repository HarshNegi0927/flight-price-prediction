import os
import warnings
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    FunctionTransformer
)

from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
    RareLabelEncoder,
    MeanEncoder,
    CountFrequencyEncoder
)

sklearn.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")

# ─── AIRLINE ──────────────────────────────────────────────────────────────────
air_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])

# ─── DATE OF JOURNEY ──────────────────────────────────────────────────────────
feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
    ("dt", DatetimeFeatures(
        features_to_extract=feature_to_extract,
        yearfirst=True,
        format="mixed"
    )),
    ("scaler", MinMaxScaler())
])

# ─── SOURCE & DESTINATION ─────────────────────────────────────────────────────
location_pipe1 = Pipeline(steps=[
    ("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)),
    ("encoder", MeanEncoder()),
    ("scaler", PowerTransformer())
])

def is_north(X):
    columns = X.columns.to_list()
    north_cities = ["Delhi", "Kolkata", "Mumbai", "New Delhi"]
    return (
        X
        .assign(**{
            f"{col}_is_north": X.loc[:, col].isin(north_cities).astype(int)
            for col in columns
        })
        .drop(columns=columns)
    )

location_transformer = FeatureUnion(transformer_list=[
    ("part1", location_pipe1),
    ("part2", FunctionTransformer(func=is_north))
])

# ─── DEP_TIME & ARRIVAL_TIME ──────────────────────────────────────────────────
time_pipe1 = Pipeline(steps=[
    ("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
    ("scaler", MinMaxScaler())
])

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
    columns = X.columns.to_list()
    X_temp = X.assign(**{
        col: pd.to_datetime(X.loc[:, col]).dt.hour
        for col in columns
    })
    return (
        X_temp
        .assign(**{
            f"{col}_part_of_day": np.select(
                [X_temp.loc[:, col].between(morning, noon, inclusive="left"),
                 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
                 X_temp.loc[:, col].between(eve, night, inclusive="left")],
                ["morning", "afternoon", "evening"],
                default="night"
            )
            for col in columns
        })
        .drop(columns=columns)
    )

time_pipe2 = Pipeline(steps=[
    ("part", FunctionTransformer(func=part_of_day)),
    ("encoder", CountFrequencyEncoder()),
    ("scaler", MinMaxScaler())
])

# FIX: peak hour flag added
def is_peak_hour(X):
    columns = X.columns.to_list()
    result = {}
    for col in columns:
        hour = pd.to_datetime(X.loc[:, col]).dt.hour
        result[f"{col}_is_peak"] = (
            hour.between(6, 9, inclusive="both") |
            hour.between(17, 20, inclusive="both")
        ).astype(int)
    return pd.DataFrame(result, index=X.index)

# FIX: part3 added
time_transformer = FeatureUnion(transformer_list=[
    ("part1", time_pipe1),
    ("part2", time_pipe2),
    ("part3", FunctionTransformer(func=is_peak_hour))
])

# ─── DURATION ─────────────────────────────────────────────────────────────────
class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None, percentiles=None, gamma=0.1):
        self.variables = variables
        # FIX: 5 anchors instead of 3
        self.percentiles = percentiles if percentiles is not None else [
            0.10, 0.25, 0.50, 0.75, 0.90
        ]
        self.gamma = gamma

    def fit(self, X, y=None):
        if not self.variables:
            self.variables = X.select_dtypes(include="number").columns.to_list()
        self.reference_values_ = {
            col: (
                X.loc[:, col]
                .quantile(self.percentiles)
                .values
                .reshape(-1, 1)
            )
            for col in self.variables
        }
        return self

    def transform(self, X):
        objects = []
        for col in self.variables:
            columns = [
                f"{col}_rbf_{int(p * 100)}"
                for p in self.percentiles
            ]
            obj = pd.DataFrame(
                data=rbf_kernel(
                    X.loc[:, [col]],
                    Y=self.reference_values_[col],
                    gamma=self.gamma
                ),
                columns=columns,
                index=X.index
            )
            objects.append(obj)
        return pd.concat(objects, axis=1)


def duration_category(X, short=180, med=400):
    return (
        X
        .assign(duration_cat=np.select(
            [X.duration.lt(short),
             X.duration.between(short, med, inclusive="left")],
            ["short", "medium"],
            default="long"
        ))
        .drop(columns="duration")
    )


def is_over(X, value=1000):
    return (
        X
        .assign(**{f"duration_over_{value}": X.duration.ge(value).astype(int)})
        .drop(columns="duration")
    )


duration_pipe1 = Pipeline(steps=[
    ("rbf", RBFPercentileSimilarity()),
    ("scaler", PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
    ("cat", FunctionTransformer(func=duration_category)),
    ("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
    ("part1", duration_pipe1),
    ("part2", duration_pipe2),
    ("part3", FunctionTransformer(func=is_over)),
    ("part4", StandardScaler())
])

duration_transformer = Pipeline(steps=[
    ("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
    ("imputer", SimpleImputer(strategy="median")),
    ("union", duration_union)
])

# ─── TOTAL_STOPS ──────────────────────────────────────────────────────────────
def is_direct(X):
    return X.assign(is_direct_flight=X.total_stops.eq(0).astype(int))

total_stops_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("direct", FunctionTransformer(func=is_direct))
])

# ─── ADDITIONAL_INFO ──────────────────────────────────────────────────────────
info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(X):
    return X.assign(additional_info=X.additional_info.ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
    ("part1", info_pipe1),
    ("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
    ("union", info_union)
])

# ─── COLUMN TRANSFORMER ───────────────────────────────────────────────────────
column_transformer = ColumnTransformer(transformers=[
    ("air",      air_transformer,         ["airline"]),
    ("doj",      doj_transformer,         ["date_of_journey"]),
    ("location", location_transformer,    ["source", "destination"]),
    ("time",     time_transformer,        ["dep_time", "arrival_time"]),
    ("dur",      duration_transformer,    ["duration"]),
    ("stops",    total_stops_transformer, ["total_stops"]),
    ("info",     info_transformer,        ["additional_info"])
], remainder="passthrough")

# ─── FEATURE SELECTOR ─────────────────────────────────────────────────────────
estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

# FIX: threshold 0.1 → 0.05
selector = SelectBySingleFeaturePerformance(
    estimator=estimator,
    scoring="r2",
    threshold=0.05
)

# ─── PREPROCESSOR ─────────────────────────────────────────────────────────────
preprocessor = Pipeline(steps=[
    ("ct",       column_transformer),
    ("selector", selector)
])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# read training data
train = pd.read_csv(os.path.join(BASE_DIR, "Data", "train.csv"))
X_train = train.drop(columns="price")
y_train = train.price.copy()

# FIX: fit on log1p(price) — same as SageMaker training
preprocessor.fit(X_train, np.log1p(y_train))
joblib.dump(preprocessor, os.path.join(BASE_DIR, "preprocessor.joblib"))

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Flight Price Prediction — AWS SageMaker")

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox(
        "Airline:",
        options=X_train.airline.unique()
    )
    source = st.selectbox(
        "Source:",
        options=X_train.source.unique()
    )
    destination = st.selectbox(
        "Destination:",
        options=X_train.destination.unique()
    )

with col2:
    doj = st.date_input("Date of Journey:")
    dep_time = st.time_input("Departure Time:")
    arrival_time = st.time_input("Arrival Time:")

with col3:
    duration = st.number_input(
        "Duration (mins):",
        step=1,
        min_value=0
    )
    total_stops = st.number_input(
        "Total Stops:",
        step=1,
        min_value=0
    )
    additional_info = st.selectbox(
        "Additional Info:",
        options=X_train.additional_info.unique()
    )

x_new = pd.DataFrame(dict(
    airline=[airline],
    date_of_journey=[doj],
    source=[source],
    destination=[destination],
    dep_time=[dep_time],
    arrival_time=[arrival_time],
    duration=[duration],
    total_stops=[total_stops],
    additional_info=[additional_info]
)).astype({
    col: "str"
    for col in ["date_of_journey", "dep_time", "arrival_time"]
})

# ─── PREDICT ──────────────────────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        saved_preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.joblib"))
        x_new_pre = saved_preprocessor.transform(x_new)

        # FIX: JSON model load karo
        model = xgb.Booster()
        model.load_model(os.path.join(BASE_DIR, "xgboost-model.json"))

        x_new_xgb = xgb.DMatrix(x_new_pre.values)
        pred_log = model.predict(x_new_xgb)[0]

        # FIX: expm1 — reverse log1p transform
        pred = np.expm1(pred_log)

        st.success(f"### Predicted Price: ₹{pred:,.0f} INR")

    except Exception as e:
        st.error(f"Error: {e}")