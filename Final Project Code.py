# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import googlemaps
import os
from dotenv import load_dotenv
import folium
from folium.plugins import HeatMap
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# File path
file_path = r'C:\Users\Server\Documents\Danielle\DS785\UStransportation_Rail_Derailment_Data.xlsx'

# Read file
df = pd.read_excel(file_path)

# Checking Data to see what needs to be cleaned
print(df.info())
print(df.describe())
print(df.nunique())

# Columns to drop
columns_to_drop = [
    "Adjunct Code 1", "Adjunct Code 2", "Adjunct Code 3", "Adjunct Name 1", "Adjunct Name 2", "Adjunct Name 3",
    "Other Railroad Company Grouping", "Other Railroad Class", "Other Railroad SMT Grouping", 
    "Other Railroad Holding Company", "Other Parent Railroad Name", "Other Parent Railroad Code", 
    "Other Railroad Code", "Grade Crossing ID", "First Car Initials", "Causing Car Initials", "First Car Number", "Signalization Code", "Signalization",
    "Contributing Accident Cause Code", "Contributing Accident Cause", "Reporting Railroad Company Grouping", "Reporting Railroad Class", "Reporting Railroad SMT Grouping", 
    "Reporting Parent Railroad Code", "Reporting Parent Railroad Code", "Reporting Railroad Holding Company", 
    "Other Railroad Company Grouping", "Other Railroad Class", "Other Railroad SMT Grouping", 
    "Other Parent Railroad Code", "Other Parent Railroad Code", "Other Parent Railroad Name", 
    "Other Railroad Holding Company", "Maintenance Railroad Company Grouping", "Maintenance Railroad Class", 
    "Maintenance Railroad SMT Grouping", "Maintenance Parent Railroad Code", "Maintenance Parent Railroad Name", 
    "Maintenance Railroad Holding Company", "Subdivision", "PDF Link", "Persons Evacuated", "Total Injured Form 54", "Total Killed Form 54", "Total Persons Injured", 
    "Total Persons Killed", "Reporting Railroad Injuries for 55a", "Reporting Railroad Fatalities for 55a", 
    "Others Injured", "Others Killed", "Passengers Injured", "Passengers Killed", "Railroad Employees Injured", 
    "Railroad Employees Killed", "Derailed Cabooses", "Special Study 1", "Special Study 2", "Derailed Empty Passenger Cars",
    "Derailed Empty Freight Cars", "Derailed Loaded Passenger Cars", "Derailed Loaded Freight Cars", "Derailed Rear End Remote Locomotives",
    "Derailed Rear End Manual Locomotives", "Derailed Mid Train Remote Locomotives", "First Car Initials", "Passengers Transported",
    "Derailed Mid Train Manual Locomotives", "Derailed Head End Locomotives", "Hazmat Released Cars", "Hazmat Cars Damaged, Hazmat Cars",
    "Causing Car Initials", "Other Railroad Name", "Method of Operation", "Remote Control Locomotive", "Remote Control Locomotive Code", 
]
df_drop = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
# Filter data for derailment accident types
df_filter = df_drop[df_drop['Accident Type'].str.lower()== 'derailment']
# Making sure Accident Year is numeric
df_filter['Accident Year'] = pd.to_numeric(df_filter['Accident Year'], errors='coerce')
# Date column is datetime format
df_filter["Date"] = pd.to_datetime(df_filter["Date"])
# Filter data for dates 1990 and forward
df_all_dr = df_filter.copy()
df_filter = df_filter[df_filter["Date"].dt.year >= 1990]
# Finding columns with 40% or more of missing data
miss_values = df_filter.isnull().sum()
miss_percent = (miss_values / len(df_filter)) * 100
miss_data = pd.DataFrame({'Missing Values': miss_values, 'Percent Missing': miss_percent})
high_miss_data = miss_data[miss_data['Percent Missing'] > 40].sort_values(by='Percent Missing', ascending=False)
print(f"Percent of Missing Values: ", high_miss_data)
# Add column "Hazmat Involvement Indicator" to indicate Hazmat involved in derailment
df_filter["Hazmat Involvement Indicator"] = (df_filter["Hazmat Cars Damaged"] > 0).astype(int)
# Check for duplicate rows
print(df_filter.duplicated())

# Convert categorical columns to numerics
categorical_columns = ["Track Class", "Train Direction Code"]

# Convert categorical columns to numerical
for col in categorical_columns:
    if col in df_filter.columns:
        df_filter[col] = pd.factorize(df_filter[col])[0]

# Exploratory analysis
# Time analysis of derailments before and after date filter
yearly_derailments = df_all_dr.groupby(df_all_dr["Date"].dt.year).size()
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_derailments.index, y=yearly_derailments.values, marker="o")
#plt.title("Historical Yearly Train Derailments Trend")
plt.xlabel("Year")
plt.ylabel("Number of Derailments")
plt.show()

yearly_derailments = df_filter.groupby(df_filter["Date"].dt.year).size()
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_derailments.index, y=yearly_derailments.values, marker="o")
#plt.title("Yearly Train Derailments Trend From 1984-2024")
plt.xlabel("Year")
plt.ylabel("Number of Derailments")
plt.show()
# Derailments per rail company
plt.figure(figsize=(12, 6))
top_companies = df_filter["Reporting Railroad Name"].value_counts().nlargest(10)
sns.barplot(x=top_companies.index, y=top_companies.values)
plt.xticks(rotation=45, ha="right", fontsize=10)
#plt.title("Top 10 Rail Companies with Most Derailments", fontsize=14)
plt.xlabel("Railroad Company", fontsize=12)
plt.ylabel("Number of Derailments", fontsize=12)
plt.tight_layout()
plt.show()
# Derailmeant per weather condition
plt.figure(figsize=(8, 5))
sns.countplot(data=df_filter, x="Weather Condition", order=df_filter["Weather Condition"].value_counts().index)
plt.xticks(rotation=45)
#plt.title("Impact of Weather Conditions on Derailments")
plt.xlabel("Weather Condition")
plt.ylabel("Number of Derailments")
plt.show()
# high count derailment causes
# high count of 1000 or more
threshold = 1000
cause_count = df_filter["Primary Accident Cause Code"].value_counts()
high_count_codes = cause_count[cause_count > threshold].index
df_high_count = df_filter[df_filter["Primary Accident Cause Code"].isin(high_count_codes)]
# plot
plt.figure(figsize=(8, 5))
sns.countplot(data=df_high_count, x="Primary Accident Cause Code", order=high_count_codes)
plt.xticks(rotation=45)
#plt.title("Total Derailment Types (Over 1000 Accounts Only)")
plt.xlabel("Derailment Cause Code")
plt.ylabel("Number of Derailments")
plt.tight_layout()
plt.show()


# Year-Month heatmap After 2010
df_filter["Year"] = df_filter["Date"].dt.year
df_filter["Month"] = df_filter["Date"].dt.month_name()
df_2010 = df_filter[df_filter["Year"] >= 2010]
# Pivot table for heatmap
monthly = df_2010.pivot_table(index="Month", columns="Year", values="Accident Number", aggfunc="count").fillna(0)
# Order months
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
monthly = monthly.reindex(month_order)
# Plot
plt.figure(figsize=(14, 6))
sns.heatmap(monthly, cmap="YlOrRd", linewidths=0.5)
#plt.title("Heatmap of Derailments by Month and Year")
plt.xlabel("Year")
plt.ylabel("Month")
plt.tight_layout()
plt.show()

# Create Hazmat Category (if it's 1/0, label it)
df_filter["Hazmat"] = df_filter["Hazmat Involvement Indicator"].map({1: "Hazmat", 0: "Non-Hazmat"})
# Pivot table of average cost by year and hazmat status
hazmat_pivot = df_filter.pivot_table(index="Hazmat", columns="Year", values="Total Damage Cost", aggfunc="mean")
# Plot heatmap
plt.figure(figsize=(14, 4))
sns.heatmap(hazmat_pivot, cmap="Reds", annot=False, linewidths=0.5, cbar_kws={'label': 'Avg Damage Cost ($)'})
#plt.title("Average Derailment Cost by Hazmat Involvement (1984–2024)")
plt.xlabel("Year")
plt.ylabel("Hazmat Status")
plt.tight_layout()
plt.show()

# Get Google Geocoding API key: 
load_dotenv(r"C:\Users\Server\Documents\Danielle\DS785\Google API key.env")
api_key = os.getenv("GOOGLE_API_KEY")
# Start Google API
gmap = googlemaps.Client(key=api_key)
# Get Lat and Long
def get_lat_lon_google(station, county, state):
    query = f"{station}, {county}, {state}, USA"
    try:
        geocode_result = gmap.geocode(query)
        if geocode_result:
            location = geocode_result[0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            return None, None
    except Exception as e:
        print(f"Error geocoding {query}: {e}")
        return None, None
# Add Lat and Longs to dataframe
for index, row in df_filter.iterrows():
    if pd.isnull(row["Latitude"]) or pd.isnull(row["Longitude"]):
        lat, lon = get_lat_lon_google(row["Station"], row["County Name"], row["State Name"])
        
        # Only update if valid coordinates were found
        if lat is not None and lon is not None:
            df_filter.at[index, "Latitude"] = lat
            df_filter.at[index, "Longitude"] = lon

        time.sleep(0.5)
# Convert lat and long to numeric
df_filter["Latitude"] = pd.to_numeric(df_filter["Latitude"], errors="coerce")
df_filter["Longitude"] = pd.to_numeric(df_filter["Longitude"], errors="coerce")

# Map Derailments
m = folium.Map(location=[39.8283, -98.5795], zoom_start=5)
heat_data = df_filter[["Latitude", "Longitude"]].dropna().values.tolist()
HeatMap(heat_data, 
        radius=12,
        blur=15,
        max_zoom=9 
       ).add_to(m)
m.save(r"C:\Users\Server\Documents\Danielle\DS785\derailment_map.html")

# Correlation analysis using heatmap
corr_matrix = df_filter.corr(numeric_only=True)
# A lot of variables filtering out for high correlation
high_corr = corr_matrix[(corr_matrix.abs() > 0.5) & (corr_matrix.abs() < 1.0)]
high_corr = high_corr.dropna(how='all').dropna(axis=1, how='all')
# plot
plt.figure(figsize=(12, 10))
sns.heatmap(high_corr, annot=False, cmap="coolwarm", linewidths=0.5)
#plt.title("Filtered Correlation Matrix")
plt.tight_layout()
plt.show()

# Regression analysis for derailment costs
regression_models = {
    "Predicting Total Damage Cost": {
        "X": ["Train Speed", "Weather Condition Code", "Gross Tonnage", "Track Class", "Temperature", "Hazmat Cars"],
        "y": "Total Damage Cost"
    },
    "Predicting Train Speed at Accident": {
        "X": ["Weather Condition Code", "Visibility Code", "Track Class", "Train Direction Code", "Gross Tonnage"],
        "y": "Train Speed"
    },
    "Predicting Equipment Damage Cost": {
        "X": ["Train Speed", "Track Class", "Loaded Freight Cars", "Empty Freight Cars"],
        "y": "Equipment Damage Cost"
    }
}
# place to store results
regression_results = {}
for model_name, params in regression_models.items():
    available_columns = [col for col in params["X"] if col in df_filter.columns]

    # Define predict and target variables
    X = df_filter[available_columns].copy()
    y = df_filter[params["y"]].copy()
    # Fill missing values with the median
    X = X.fillna(X.median())
    y = y.fillna(y.median())
    # Add a constant term for the intercept
    X = sm.add_constant(X)
    # Perform regression analysis
    model = sm.OLS(y, X).fit()
    # Store results
    regression_results[model_name] = model.summary()
# Display results
print(regression_results)

# Predictive Modeling
df = df_filter.copy()
# Define the target variable Binary Classification: High vs. Low Damage Cost
threshold = df["Total Damage Cost"].quantile(0.75)
df["High_Cost"] = (df["Total Damage Cost"] > threshold).astype(int)

# Select independent variables
features = ["Train Speed", "Weather Condition Code", "Gross Tonnage", "Track Class", "Temperature", "Hazmat Cars"]
X = df[features]
y = df["High_Cost"]

# Convert categorical variables into numerical encoding
for col in ["Track Class"]:
    if col in X.columns:
        X[col] = pd.factorize(X[col])[0]

# Fill missing values with median
X = X.fillna(X.median())
feature_names = X.columns.tolist()
# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"Training size: {X_train.shape}, Testing size: {X_test.shape}")

# Initialize the model
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Get feature importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Create importance DataFrame
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": result.importances_mean
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature")
#plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()

# Example new derailment conditions
new_data = pd.DataFrame({
    "Train Speed": [45],  
    "Weather Condition Code": [2],  
    "Gross Tonnage": [12000],
    "Track Class": [1],  
    "Temperature": [50],
    "Hazmat Cars": [5]
})

# Convert categorical variables
new_data["Track Class"] = pd.factorize(new_data["Track Class"])[0]

# Standardize features
new_data_scaled = scaler.transform(new_data)

# Make prediction
prediction = model.predict(new_data_scaled)
print("Prediction (1=High Cost, 0=Low Cost):", prediction[0])


# Define target variable (1 = High-Cost Derailment, 0 = Low-Cost)
df["High_Cost"] = (df["Total Damage Cost"] > df["Total Damage Cost"].median()).astype(int)

# Select features
features = ["Train Speed", "Weather Condition Code", "Gross Tonnage", "Track Class", "Temperature", "Hazmat Cars"]
X = df[features]
y = df["High_Cost"]

# Convert categorical variables
X["Track Class"] = pd.factorize(X["Track Class"])[0]

# Handle missing values
X = X.fillna(X.median())

# Standardize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Train a Random Forest Classifier
model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"AUC Score: {auc_score:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("ROC Curve for Derailment Probability Prediction")
plt.legend()
plt.show()
new_accident = pd.DataFrame({
    "Train Speed": [45],  
    "Weather Condition Code": [2],  
    "Gross Tonnage": [12000],
    "Track Class": [1],   
    "Temperature": [50],
    "Hazmat Cars": [5]
})

# Convert categorical variable
new_accident["Track Class"] = pd.factorize(new_accident["Track Class"])[0]

# Standardize features
new_accident_scaled = scaler.transform(new_accident)

# Predict probability
probability = model.predict_proba(new_accident_scaled)[:, 1]
print(f"Predicted Probability of High-Cost Derailment: {probability[0]:.2f}")

