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

# Sidebar Filters
st.sidebar.header("🔎 Filter Options")

gender = st.sidebar.selectbox("Select Gender", options=["All"] + list(df["Sex"].unique()))
pclass = st.sidebar.selectbox("Select Passenger Class", options=["All"] + sorted(df["Pclass"].unique()))
embarked = st.sidebar.selectbox("Select Embarked Port", options=["All"] + list(df["Embarked"].dropna().unique()))
survival = st.sidebar.selectbox("Select Survival Status", options=["All", 0, 1])
age_min = int(df["Age"].min())
age_max = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

# Apply filters
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]

if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

if embarked != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == embarked]

if survival != "All":
    filtered_df = filtered_df[filtered_df["Survived"] == survival]

filtered_df = filtered_df[(filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])]

# Show Filtered Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)

# Filtered Data Preview
st.subheader("Filtered Data Preview")
st.write(filtered_df.head())

# ======= 3×3 Grid Visualizations =======

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("### Age Distribution by Survival")
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    sns.histplot(data=filtered_df, x="Age", hue="Survived", multiple="stack", bins=20, ax=ax2)
    st.pyplot(fig2)

with col3:
    st.markdown("### Survival Rate by Passenger Class")
    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", estimator=lambda x: sum(x)/len(x), ax=ax3)
    ax3.set_ylabel("Survival Rate")
    st.pyplot(fig3)

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### Fare Distribution by Class")
    fig4, ax4 = plt.subplots(figsize=(5, 3.5))
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax4)
    st.pyplot(fig4)

with col5:
    st.markdown("### Correlation Heatmap")
    corr = filtered_df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
    fig5, ax5 = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

with col6:
    st.markdown("### Embarked Port Count")
    fig6, ax6 = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=filtered_df, x="Embarked", ax=ax6)
    st.pyplot(fig6)

# Row 3
col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("### Age vs Fare (Scatter Plot)")
    fig7, ax7 = plt.subplots(figsize=(5, 3.5))
    sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Survived", ax=ax7)
    st.pyplot(fig7)

with col8:
    st.markdown("### Sibling/Spouse Count Distribution")
    fig8, ax8 = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=filtered_df, x="SibSp", ax=ax8)
    st.pyplot(fig8)

with col9:
    st.markdown("### Parent/Children Count Distribution")
    fig9, ax9 = plt.subplots(figsize=(5, 3.5))
    sns.countplot(data=filtered_df, x="Parch", ax=ax9)
    st.pyplot(fig9)
