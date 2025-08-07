# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Title
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=df["Pclass"].unique())

# Apply filters
filtered_df = df[(df["Sex"] == gender) & (df["Pclass"] == pclass)]

# Filtered Data Preview
st.subheader("Filtered Data Preview")
st.write(filtered_df.head())

# === Visualizations ===

# 1. Survival Count by Gender
st.subheader("Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
st.pyplot(fig1)

# 2. Age Distribution by Survival
with st.expander("📊 Age Distribution by Survival"):
    fig2, ax2 = plt.subplots()
    sns.histplot(data=filtered_df, x="Age", hue="Survived", multiple="stack", bins=20, ax=ax2)
    st.pyplot(fig2)

# 3. Survival Rate by Passenger Class
with st.expander("📈 Survival Rate by Passenger Class"):
    fig3, ax3 = plt.subplots()
    sns.barplot(data=df, x="Pclass", y="Survived", estimator=lambda x: sum(x)/len(x), ax=ax3)
    ax3.set_ylabel("Survival Rate")
    st.pyplot(fig3)

# 4. Fare Distribution by Class
with st.expander("💰 Fare Distribution by Class"):
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax4)
    st.pyplot(fig4)

# 5. Correlation Heatmap
with st.expander("🔍 Correlation Heatmap"):
    corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
    fig5, ax5 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

# 6. Embarked Location Count
with st.expander("🗺️ Passenger Count by Embarkation Port"):
    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", ax=ax6)
    st.pyplot(fig6)
