"""
Nassau Candy Distributor — Factory Reallocation & Shipping Optimizer
ALL-IN-ONE FILE (data prep + modeling + clustering + simulation + optimization + Streamlit dashboard)

HOW TO RUN
-----------
Run the analysis pipeline (prints results, saves files):
    python nassau_candy_all_in_one.py

Launch the interactive dashboard:
    streamlit run nassau_candy_all_in_one.py

Both modes use this same single file - no other project files needed.
"""

import os
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

import numpy as np
import pandas as pd
import joblib
import zipcodes
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans


# ======================================================================
# SECTION 1: CONFIG  (reference data: factories, product map, assumptions)
# ======================================================================

# ---- Factory coordinates ----
FACTORY_COORDS = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

FACTORIES = list(FACTORY_COORDS.keys())

# ---- Product -> current (assigned) factory ----
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

PRODUCT_DIVISION = {
    "Wonka Bar - Nutty Crunch Surprise": "Chocolate",
    "Wonka Bar - Fudge Mallows": "Chocolate",
    "Wonka Bar -Scrumdiddlyumptious": "Chocolate",
    "Wonka Bar - Milk Chocolate": "Chocolate",
    "Wonka Bar - Triple Dazzle Caramel": "Chocolate",
    "Laffy Taffy": "Sugar",
    "SweeTARTS": "Sugar",
    "Nerds": "Sugar",
    "Fun Dip": "Sugar",
    "Fizzy Lifting Drinks": "Other",
    "Everlasting Gobstopper": "Sugar",
    "Hair Toffee": "Sugar",
    "Lickable Wallpaper": "Other",
    "Wonka Gum": "Other",
    "Kazookles": "Other",
}

# ---- Canadian city centroids (postal-code-free fallback) ----
CA_CITY_COORDS = {
    "Toronto": (43.6532, -79.3832),
    "Montreal": (45.5019, -73.5674),
    "Vancouver": (49.2827, -123.1207),
    "Calgary": (51.0447, -114.0719),
    "Quebec City": (46.8139, -71.2080),
    "Winnipeg": (49.8951, -97.1384),
    "Charlottetown": (46.2382, -63.1311),
    "Edmonton": (53.5461, -113.4938),
    "Moncton": (46.0878, -64.7782),
    "Halifax": (44.6488, -63.5752),
    "St. John's": (47.5615, -52.7126),
    "Regina": (50.4452, -104.6189),
}

# ---- Ship Mode base transit assumptions (days), used to reconstruct a
# realistic lead-time target since Order Date / Ship Date in the source
# file are corrupted (see README "Known data issue"). ----
SHIP_MODE_BASE_DAYS = {
    "Same Day": 0,
    "First Class": 2,
    "Second Class": 3,
    "Standard Class": 5,
}
MILES_PER_EXTRA_DAY = 600  # +1 day of transit per this many miles

RAW_DATA_PATH = "data/raw_data.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"
MODEL_PATH = "models/lead_time_model.joblib"
ENCODERS_PATH = "models/encoders.joblib"
CLUSTER_PATH = "models/route_clusters.joblib"
RECOMMENDATIONS_PATH = "outputs/factory_recommendations.csv"

RANDOM_STATE = 42


# ======================================================================
# SECTION 2: DATA PREP  (clean, geocode, compute distance, reconstruct lead time)
# ======================================================================

import numpy as np
import pandas as pd
import zipcodes



def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance in miles between two lat/long points (vectorized)."""
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def load_raw(path=RAW_DATA_PATH):
    df = pd.read_csv(path, dtype={"Postal Code": str})
    return df


def clean_postal_codes(df):
    df = df.copy()
    df["Postal Code"] = df["Postal Code"].str.strip()
    mask_us = df["Country/Region"] == "United States"
    df.loc[mask_us, "Postal Code"] = df.loc[mask_us, "Postal Code"].str.zfill(5)
    return df


def geocode_customers(df):
    """Adds Customer Lat / Customer Long columns.
    US rows: looked up offline via the `zipcodes` package (bundled data,
    no network call). Canada rows: mapped via a small city centroid table
    since the dataset only contains 12 distinct Canadian cities.
    """
    df = df.copy()
    zip_cache = {}

    def us_latlong(z):
        if z not in zip_cache:
            r = zipcodes.matching(z)
            zip_cache[z] = (float(r[0]["lat"]), float(r[0]["long"])) if r else (np.nan, np.nan)
        return zip_cache[z]

    lats, longs = [], []
    for _, row in df.iterrows():
        if row["Country/Region"] == "United States":
            lat, lon = us_latlong(row["Postal Code"])
        else:
            lat, lon = CA_CITY_COORDS.get(row["City"], (np.nan, np.nan))
        lats.append(lat)
        longs.append(lon)

    df["Customer Lat"] = lats
    df["Customer Long"] = longs
    n_missing = df["Customer Lat"].isna().sum()
    if n_missing:
        print(f"[data_prep] WARNING: {n_missing} rows could not be geocoded and were dropped.")
        df = df.dropna(subset=["Customer Lat", "Customer Long"]).reset_index(drop=True)
    return df


def attach_factory_info(df):
    df = df.copy()
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    unmapped = df[df["Factory"].isna()]["Product Name"].unique()
    if len(unmapped):
        print(f"[data_prep] WARNING: unmapped products dropped: {list(unmapped)}")
        df = df.dropna(subset=["Factory"]).reset_index(drop=True)
    _factory_lat_lookup = {k: v[0] for k, v in FACTORY_COORDS.items()}
    _factory_long_lookup = {k: v[1] for k, v in FACTORY_COORDS.items()}
    df["Factory Lat"] = df["Factory"].map(_factory_lat_lookup)
    df["Factory Long"] = df["Factory"].map(_factory_long_lookup)
    return df


def compute_distance(df):
    df = df.copy()
    df["Shipping Distance (mi)"] = haversine_miles(
        df["Customer Lat"], df["Customer Long"], df["Factory Lat"], df["Factory Long"]
    )
    return df


def synthesize_lead_time(df, random_state=RANDOM_STATE):
    """Reconstructs a realistic lead time in days from Ship Mode + distance.
    See module docstring for why this replaces the corrupted raw dates.
    """
    df = df.copy()
    rng = np.random.default_rng(random_state)
    base = df["Ship Mode"].map(SHIP_MODE_BASE_DAYS).fillna(5)
    distance_days = df["Shipping Distance (mi)"] / MILES_PER_EXTRA_DAY
    jitter = rng.normal(0, 0.6, size=len(df))
    lead_time = (base + distance_days + jitter).clip(lower=0)
    df["Lead Time (days)"] = lead_time.round().astype(int)
    return df


def add_financials(df):
    df = df.copy()
    df["Profit Margin"] = df["Gross Profit"] / df["Sales"]
    return df


def cap_outliers(df, cols=("Sales", "Profit Margin"), k=1.5):
    """Winsorizes (caps, does not drop) outliers using the IQR rule."""
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def data_prep_run_pipeline(raw_path=RAW_DATA_PATH, save_path=PROCESSED_DATA_PATH):
    df = load_raw(raw_path)
    df = clean_postal_codes(df)
    df = geocode_customers(df)
    df = attach_factory_info(df)
    df = compute_distance(df)
    df = synthesize_lead_time(df)
    df = add_financials(df)
    df = cap_outliers(df)
    df.to_csv(save_path, index=False)
    print(f"[data_prep] Saved processed data: {save_path}  shape={df.shape}")
    return df


# ======================================================================
# SECTION 3: MODELING  (Linear Regression / Random Forest / Gradient Boosting)
# ======================================================================

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


FEATURES_NUMERIC = ["Shipping Distance (mi)", "Units", "Sales"]
FEATURES_CATEGORICAL = ["Ship Mode", "Region", "Factory", "Division"]
TARGET = "Lead Time (days)"


def build_feature_matrix(df, encoders=None, fit=True):
    """Encodes categorical features with LabelEncoder and scales numerics.
    Returns X (np.array), y (np.array or None), and the encoders dict.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    cat_encoded = pd.DataFrame(index=df.index)
    for col in FEATURES_CATEGORICAL:
        if fit:
            le = LabelEncoder()
            cat_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # handle unseen categories gracefully by mapping to a known class
            known = set(le.classes_)
            safe_vals = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            cat_encoded[col] = le.transform(safe_vals)

    num = df[FEATURES_NUMERIC].reset_index(drop=True)
    cat_encoded = cat_encoded.reset_index(drop=True)

    if fit:
        scaler = StandardScaler()
        num_scaled = pd.DataFrame(scaler.fit_transform(num), columns=FEATURES_NUMERIC)
        encoders["scaler"] = scaler
    else:
        scaler = encoders["scaler"]
        num_scaled = pd.DataFrame(scaler.transform(num), columns=FEATURES_NUMERIC)

    X = pd.concat([num_scaled, cat_encoded], axis=1)
    y = df[TARGET].values if TARGET in df.columns else None
    return X, y, encoders


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def train_all_models(X_train, y_train, X_test, y_test):
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08, random_state=RANDOM_STATE
        ),
    }
    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        fitted[name] = model
        print(f"[modeling] {name:18s} RMSE={metrics['RMSE']:.3f}  MAE={metrics['MAE']:.3f}  R2={metrics['R2']:.3f}")
    return results, fitted


def select_best(results, fitted):
    best_name = min(results, key=lambda n: results[n]["RMSE"])
    print(f"[modeling] Best model: {best_name}")
    return best_name, fitted[best_name]


def modeling_run_pipeline(data_path=PROCESSED_DATA_PATH):
    df = pd.read_csv(data_path)
    X, y, encoders = build_feature_matrix(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    results, fitted = train_all_models(X_train, y_train, X_test, y_test)
    best_name, best_model = select_best(results, fitted)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"[modeling] Saved best model ({best_name}) -> {MODEL_PATH}")

    results_df = pd.DataFrame(results).T.sort_values("RMSE")
    results_df.to_csv("outputs/model_comparison.csv")
    return best_name, best_model, encoders, results_df


# ======================================================================
# SECTION 4: CLUSTERING  (route performance clusters)
# ======================================================================

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROUTE_FEATURES = ["Shipping Distance (mi)", "Lead Time (days)", "Profit Margin", "orders"]


def build_route_table(df):
    """Aggregates to one row per (Region, Factory) route."""
    route = df.groupby(["Region", "Factory"]).agg(
        orders=("Order ID", "count"),
        **{
            "Shipping Distance (mi)": ("Shipping Distance (mi)", "mean"),
            "Lead Time (days)": ("Lead Time (days)", "mean"),
            "Profit Margin": ("Profit Margin", "mean"),
        }
    ).reset_index()
    return route


def cluster_routes(route_df, n_clusters=3, random_state=RANDOM_STATE):
    scaler = StandardScaler()
    X = scaler.fit_transform(route_df[ROUTE_FEATURES])
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    route_df = route_df.copy()
    route_df["Cluster"] = km.fit_predict(X)

    # Label clusters by their average lead time (higher = slower/riskier)
    cluster_lead = route_df.groupby("Cluster")["Lead Time (days)"].mean().sort_values()
    label_map = {}
    labels = ["Fast / Efficient", "Moderate", "Slow / Congested"]
    for rank, (cluster_id, _) in enumerate(cluster_lead.items()):
        label_map[cluster_id] = labels[min(rank, len(labels) - 1)]
    route_df["Cluster Label"] = route_df["Cluster"].map(label_map)

    return route_df, km, scaler


def clustering_run_pipeline(data_path=PROCESSED_DATA_PATH, n_clusters=3):
    df = pd.read_csv(data_path)
    route_df = build_route_table(df)
    route_df, km, scaler = cluster_routes(route_df, n_clusters=n_clusters)
    route_df = route_df.sort_values(["Cluster Label", "Lead Time (days)"], ascending=[True, False])
    route_df.to_csv("outputs/route_clusters.csv", index=False)
    joblib.dump({"model": km, "scaler": scaler}, CLUSTER_PATH)
    print("[clustering] Route clusters:")
    print(route_df[["Region", "Factory", "orders", "Shipping Distance (mi)",
                     "Lead Time (days)", "Profit Margin", "Cluster Label"]].to_string(index=False))
    print("[clustering] Saved outputs/route_clusters.csv")
    return route_df


# ======================================================================
# SECTION 5: SIMULATION ENGINE  (reassignment scenarios)
# ======================================================================

import numpy as np
import pandas as pd


ASSUMED_COST_PER_MILE_PER_UNIT = 0.004  # illustrative logistics-cost proxy


def simulate_product_reassignment(df, model, encoders, product_name):
    """Returns a DataFrame: one row per (customer-region avg) x candidate
    factory, with predicted lead time, distance, and estimated impact.
    """
    prod_rows = df[df["Product Name"] == product_name].copy()
    if prod_rows.empty:
        raise ValueError(f"No historical orders found for product '{product_name}'")

    current_factory = prod_rows["Factory"].iloc[0]
    division = prod_rows["Division"].iloc[0] if "Division" in prod_rows.columns else PRODUCT_DIVISION.get(product_name)

    # Aggregate customer demand by Region for this product (centroid approach)
    region_groups = prod_rows.groupby("Region").agg(
        orders=("Order ID", "count"),
        avg_units=("Units", "mean"),
        avg_sales=("Sales", "mean"),
        avg_margin=("Profit Margin", "mean"),
        cust_lat=("Customer Lat", "mean"),
        cust_long=("Customer Long", "mean"),
        ship_mode=("Ship Mode", lambda s: s.mode().iloc[0]),
    ).reset_index()

    records = []
    for _, region_row in region_groups.iterrows():
        current_dist = haversine_miles(
            region_row["cust_lat"], region_row["cust_long"],
            *FACTORY_COORDS[current_factory]
        )
        for factory in FACTORIES:
            dist = haversine_miles(
                region_row["cust_lat"], region_row["cust_long"],
                *FACTORY_COORDS[factory]
            )
            sim_row = pd.DataFrame([{
                "Shipping Distance (mi)": dist,
                "Units": region_row["avg_units"],
                "Sales": region_row["avg_sales"],
                "Ship Mode": region_row["ship_mode"],
                "Region": region_row["Region"],
                "Factory": factory,
                "Division": division,
            }])
            X, _, _ = build_feature_matrix(sim_row, encoders=encoders, fit=False)
            pred_lead_time = float(model.predict(X)[0])

            records.append({
                "Product": product_name,
                "Region": region_row["Region"],
                "Historical Orders": int(region_row["orders"]),
                "Candidate Factory": factory,
                "Is Current Factory": factory == current_factory,
                "Predicted Lead Time (days)": round(pred_lead_time, 2),
                "Distance (mi)": round(dist, 1),
                "Current Distance (mi)": round(current_dist, 1),
                "Distance Change (%)": round(100 * (dist - current_dist) / max(current_dist, 1), 1),
                "Avg Margin (historical)": round(region_row["avg_margin"], 3),
            })

    result = pd.DataFrame(records)

    # Lead time reduction vs current factory, per region
    baseline = result[result["Is Current Factory"]][["Region", "Predicted Lead Time (days)"]]
    baseline = baseline.rename(columns={"Predicted Lead Time (days)": "Baseline Lead Time"})
    result = result.merge(baseline, on="Region", how="left")
    result["Lead Time Reduction (%)"] = round(
        100 * (result["Baseline Lead Time"] - result["Predicted Lead Time (days)"]) / result["Baseline Lead Time"], 1
    )

    # Profit impact proxy: estimated extra/reduced logistics cost per unit if
    # a $/mile-per-unit shipping cost were applied
    result["Est. Profit Impact ($/unit)"] = round(
        -1 * result["Distance Change (%)"] / 100 * result["Current Distance (mi)"] * ASSUMED_COST_PER_MILE_PER_UNIT, 3
    )

    # Scenario confidence: scaled 0-1 by historical order volume (capped)
    result["Scenario Confidence"] = (result["Historical Orders"] / result["Historical Orders"].max()).round(2)

    return result.sort_values(["Region", "Predicted Lead Time (days)"])


def run_all_products(data_path=PROCESSED_DATA_PATH):
    import joblib
    df = pd.read_csv(data_path)
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)

    all_results = []
    for product in df["Product Name"].unique():
        try:
            res = simulate_product_reassignment(df, model, encoders, product)
            all_results.append(res)
        except ValueError:
            continue

    full = pd.concat(all_results, ignore_index=True)
    full.to_csv("outputs/scenario_simulations.csv", index=False)
    print(f"[simulation] Simulated {full['Product'].nunique()} products x "
          f"{len(FACTORIES)} factories = {len(full)} scenarios")
    print("[simulation] Saved outputs/scenario_simulations.csv")
    return full


# ======================================================================
# SECTION 6: OPTIMIZATION  (scoring + top-N recommendations)
# ======================================================================

import pandas as pd


DEFAULT_WEIGHTS = {"speed": 0.5, "profit": 0.3, "risk": 0.2}


def score_scenarios(sim_df, speed_weight=0.5, profit_weight=0.3, risk_weight=0.2):
    """Adds a composite 'Recommendation Score' column (0-100, higher=better).
    speed_weight / profit_weight / risk_weight should sum to ~1.0; they are
    driven by the dashboard's "optimization priority slider".
    """
    df = sim_df.copy()

    def normalize(s, higher_is_better=True):
        rng = s.max() - s.min()
        if rng == 0:
            return pd.Series(0.5, index=s.index)
        norm = (s - s.min()) / rng
        return norm if higher_is_better else 1 - norm

    df["_speed_norm"] = normalize(df["Lead Time Reduction (%)"], higher_is_better=True)
    df["_profit_norm"] = normalize(df["Est. Profit Impact ($/unit)"], higher_is_better=True)
    # risk proxy: prefer lower distance change magnitude & higher historical confidence
    df["_risk_norm"] = normalize(
        df["Scenario Confidence"] - (df["Distance Change (%)"].abs() / 100), higher_is_better=True
    )

    df["Recommendation Score"] = (
        speed_weight * df["_speed_norm"]
        + profit_weight * df["_profit_norm"]
        + risk_weight * df["_risk_norm"]
    ) * 100
    df["Recommendation Score"] = df["Recommendation Score"].round(1)
    df = df.drop(columns=["_speed_norm", "_profit_norm", "_risk_norm"])
    return df


def top_n_recommendations(scored_df, n=3, exclude_current=True):
    """Per Product+Region, returns the top-N ranked candidate factories."""
    df = scored_df.copy()
    if exclude_current:
        df = df[~df["Is Current Factory"]]
    df = df.sort_values(["Product", "Region", "Recommendation Score"], ascending=[True, True, False])
    top = df.groupby(["Product", "Region"]).head(n).reset_index(drop=True)
    top["Rank"] = top.groupby(["Product", "Region"]).cumcount() + 1
    return top


def flag_high_risk(scored_df, distance_increase_threshold=15, profit_drop_threshold=-0.5):
    """Flags reassignment options that would make things worse: distance
    increases beyond threshold% or profit impact is meaningfully negative.
    """
    df = scored_df.copy()
    df["Risk Flag"] = (
        (df["Distance Change (%)"] > distance_increase_threshold)
        | (df["Est. Profit Impact ($/unit)"] < profit_drop_threshold)
    )
    return df


def optimization_run_pipeline(sim_path="outputs/scenario_simulations.csv",
                  speed_weight=0.5, profit_weight=0.3, risk_weight=0.2, n=3):
    sim_df = pd.read_csv(sim_path)
    scored = score_scenarios(sim_df, speed_weight, profit_weight, risk_weight)
    scored = flag_high_risk(scored)
    scored.to_csv("outputs/scored_scenarios.csv", index=False)

    top = top_n_recommendations(scored, n=n)
    top.to_csv(RECOMMENDATIONS_PATH, index=False)

    print(f"[optimization] Weights -> speed={speed_weight}, profit={profit_weight}, risk={risk_weight}")
    print(f"[optimization] Saved top-{n} recommendations -> {RECOMMENDATIONS_PATH}")
    print(top[["Product", "Region", "Rank", "Candidate Factory", "Recommendation Score",
               "Lead Time Reduction (%)", "Est. Profit Impact ($/unit)", "Risk Flag"]].head(15).to_string(index=False))
    return top, scored


# ======================================================================
# SECTION 7: FULL PIPELINE RUNNER
# ======================================================================
def run_full_pipeline():
    print("=" * 60)
    print("STEP 1/5: Data preparation")
    print("=" * 60)
    data_prep_run_pipeline()

    print("\n" + "=" * 60)
    print("STEP 2/5: Predictive modeling")
    print("=" * 60)
    modeling_run_pipeline()

    print("\n" + "=" * 60)
    print("STEP 3/5: Route clustering")
    print("=" * 60)
    clustering_run_pipeline()

    print("\n" + "=" * 60)
    print("STEP 4/5: Scenario simulation")
    print("=" * 60)
    run_all_products()

    print("\n" + "=" * 60)
    print("STEP 5/5: Optimization & recommendations")
    print("=" * 60)
    optimization_run_pipeline()

    print("\n✅ Pipeline complete. Run `streamlit run nassau_candy_all_in_one.py` to launch the dashboard.")

# ======================================================================
# SECTION 8: STREAMLIT DASHBOARD
# ======================================================================
import plotly.express as px
import streamlit as st

def run_dashboard():
    st.set_page_config(page_title="Nassau Candy Factory Optimizer", layout="wide")

    # ---- First-run bootstrap: build the pipeline outputs if they don't exist yet ----
    # (a fresh deploy on Streamlit Cloud has only the raw CSV, nothing computed)
    required_files = [
        PROCESSED_DATA_PATH,
        MODEL_PATH,
        ENCODERS_PATH,
        "outputs/scenario_simulations.csv",
        RECOMMENDATIONS_PATH,
    ]
    if not all(os.path.exists(p) for p in required_files):
        if not os.path.exists(RAW_DATA_PATH):
            st.error(
                f"Raw data file not found at `{RAW_DATA_PATH}`. Make sure it's committed "
                "to the repo at that exact path (e.g. a `data/` folder containing `raw_data.csv`)."
            )
            st.stop()
        with st.spinner("First run: preparing data, training models, and building recommendations... this takes a minute or two."):
            run_full_pipeline()
        st.rerun()

    @st.cache_data
    def load_data():
        return pd.read_csv(PROCESSED_DATA_PATH)


    @st.cache_resource
    def load_model():
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        return model, encoders


    @st.cache_data
    def load_all_scenarios():
        return pd.read_csv("outputs/scenario_simulations.csv")


    df = load_data()
    model, encoders = load_model()

    st.title("🍬 Nassau Candy — Factory Reallocation & Shipping Optimizer")
    st.caption(
        "Decision-intelligence dashboard: simulates shipping outcomes under different "
        "factory assignments and recommends reallocation to improve lead time without "
        "sacrificing profitability."
    )

    with st.expander("⚠️ Data note", expanded=False):
        st.markdown(
            "The source file's Order/Ship Date columns are corrupted (Ship Mode shows "
            "no separation in raw lead time). Lead time used throughout this app is "
            "**reconstructed** from Ship Mode + shipping distance — see `data_prep.py` "
            "docstring for details. Replace with true dates once the source is fixed."
        )

    # ---------------- Sidebar controls ----------------
    st.sidebar.header("Controls")
    product = st.sidebar.selectbox("Product", sorted(df["Product Name"].unique()))
    region_options = ["All"] + sorted(df["Region"].unique().tolist())
    region_filter = st.sidebar.selectbox("Region filter", region_options)
    ship_mode_options = ["All"] + sorted(df["Ship Mode"].unique().tolist())
    ship_mode_filter = st.sidebar.selectbox("Ship Mode filter", ship_mode_options)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Optimization priority")
    priority = st.sidebar.slider(
        "Speed  ⟷  Profit", min_value=0, max_value=100, value=50,
        help="0 = optimize purely for profit, 100 = optimize purely for speed"
    )
    speed_weight = priority / 100
    profit_weight = (100 - priority) / 100 * 0.6
    risk_weight = (100 - priority) / 100 * 0.4

    tab1, tab2, tab3, tab4 = st.tabs([
        "🏭 Factory Optimization Simulator",
        "🔀 What-If Scenario Analysis",
        "🏆 Recommendation Dashboard",
        "⚠️ Risk & Impact Panel",
    ])

    # ---------------- Tab 1: Factory Optimization Simulator ----------------
    with tab1:
        st.subheader(f"Predicted performance across factories — {product}")
        sim = simulate_product_reassignment(df, model, encoders, product)
        if region_filter != "All":
            sim = sim[sim["Region"] == region_filter]

        current_factory = df[df["Product Name"] == product]["Factory"].iloc[0]
        st.markdown(f"**Current assigned factory:** `{current_factory}`")

        fig = px.bar(
            sim, x="Candidate Factory", y="Predicted Lead Time (days)", color="Region",
            barmode="group", title="Predicted Lead Time by Candidate Factory & Region",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(
            sim, x="Candidate Factory", y="Distance (mi)", color="Region",
            barmode="group", title="Shipping Distance by Candidate Factory & Region",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(sim, use_container_width=True)

    # ---------------- Tab 2: What-If Scenario Analysis ----------------
    with tab2:
        st.subheader("Current vs Recommended Assignment")
        scored = score_scenarios(sim, speed_weight, profit_weight, risk_weight)
        best_per_region = (
            scored[~scored["Is Current Factory"]]
            .sort_values("Recommendation Score", ascending=False)
            .groupby("Region")
            .head(1)
        )
        current_rows = scored[scored["Is Current Factory"]]

        compare = pd.concat([
            current_rows.assign(Assignment="Current"),
            best_per_region.assign(Assignment="Recommended"),
        ])

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.bar(
                compare, x="Region", y="Predicted Lead Time (days)", color="Assignment",
                barmode="group", title="Lead Time: Current vs Recommended",
            )
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            fig4 = px.bar(
                compare, x="Region", y="Est. Profit Impact ($/unit)", color="Assignment",
                barmode="group", title="Profit Impact: Current vs Recommended",
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(
            compare[["Region", "Assignment", "Candidate Factory", "Predicted Lead Time (days)",
                     "Distance (mi)", "Lead Time Reduction (%)", "Est. Profit Impact ($/unit)",
                     "Recommendation Score"]],
            use_container_width=True,
        )

    # ---------------- Tab 3: Recommendation Dashboard ----------------
    with tab3:
        st.subheader("Ranked Reassignment Suggestions (all products)")
        all_sim = load_all_scenarios()
        if region_filter != "All":
            all_sim = all_sim[all_sim["Region"] == region_filter]

        all_scored = score_scenarios(all_sim, speed_weight, profit_weight, risk_weight)
        all_scored = flag_high_risk(all_scored)
        top = top_n_recommendations(all_scored, n=1)  # best alternative per product/region

        st.markdown(
            f"**Top recommendation per product × region** "
            f"(priority: {speed_weight:.0%} speed / {profit_weight+risk_weight:.0%} profit+risk)"
        )
        st.dataframe(
            top[["Product", "Region", "Candidate Factory", "Recommendation Score",
                 "Lead Time Reduction (%)", "Distance Change (%)",
                 "Est. Profit Impact ($/unit)", "Scenario Confidence", "Risk Flag"]]
            .sort_values("Recommendation Score", ascending=False),
            use_container_width=True,
        )

        fig5 = px.bar(
            top.sort_values("Recommendation Score", ascending=False).head(15),
            x="Recommendation Score", y="Product", color="Candidate Factory",
            orientation="h", title="Top 15 Recommendations by Score",
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ---------------- Tab 4: Risk & Impact Panel ----------------
    with tab4:
        st.subheader("Profit Impact Alerts & High-Risk Warnings")
        all_sim = load_all_scenarios()
        all_scored = score_scenarios(all_sim, speed_weight, profit_weight, risk_weight)
        all_scored = flag_high_risk(all_scored)

        risky = all_scored[all_scored["Risk Flag"] & ~all_scored["Is Current Factory"]]
        st.metric("High-risk reassignment scenarios flagged", len(risky))

        st.dataframe(
            risky[["Product", "Region", "Candidate Factory", "Distance Change (%)",
                   "Est. Profit Impact ($/unit)", "Scenario Confidence"]]
            .sort_values("Est. Profit Impact ($/unit)"),
            use_container_width=True,
        )

        fig6 = px.scatter(
            all_scored, x="Distance Change (%)", y="Est. Profit Impact ($/unit)",
            color="Risk Flag", hover_data=["Product", "Region", "Candidate Factory"],
            title="Risk Map: Distance Change vs Profit Impact",
        )
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")
    st.caption("Nassau Candy Distributor — Factory Reallocation & Shipping Optimization Recommendation System")

# ======================================================================
# ENTRY POINT
# ======================================================================
def _running_under_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_under_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    run_full_pipeline()
    print("\nTip: run  streamlit run nassau_candy_all_in_one.py  to open the interactive dashboard.")
