import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# --- Background Image CSS ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://raw.githubusercontent.com/nimesh455/titanic/main/titanic/titanic.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Title ---
st.markdown("""
<div style="background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 10px;">
    <h1 style="text-align: center;">🚢 Titanic Data Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# --- Load Data ---
df = pd.read_csv("cleaned_titanic.csv")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=["All"] + list(df["Sex"].unique()))
pclass = st.sidebar.selectbox("Select Passenger Class", options=["All"] + sorted(df["Pclass"].unique()))
embarked = st.sidebar.selectbox("Select Embarkation Port", options=["All"] + list(df["Embarked"].dropna().unique()))
survived = st.sidebar.selectbox("Select Survival Status", options=["All", 0, 1])
age_slider = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

# --- Apply Filters ---
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]

if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

if embarked != "All":
    filtered_df = filtered_df[filtered_df["Embarked"] == embarked]

if survived != "All":
    filtered_df = filtered_df[filtered_df["Survived"] == survived]

filtered_df = filtered_df[(filtered_df["Age"] >= age_slider[0]) & (filtered_df["Age"] <= age_slider[1])]

# --- Show Raw Data ---
if st.checkbox("Show Raw Filtered Data"):
    st.dataframe(filtered_df)

# --- Grid of Visuals (3x3) ---
st.markdown("## 📊 Visual Analysis")

# Row 1
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Survival Count by Gender")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax)
    st.pyplot(fig)

with col2:
    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x="Age", bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with col3:
    st.markdown("### Passenger Class Count")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Pclass", hue="Survived", ax=ax)
    st.pyplot(fig)

# Row 2
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### Fare Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x="Fare", bins=20, kde=True, ax=ax)
    st.pyplot(fig)

with col5:
    st.markdown("### Survival by Embarked")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", hue="Survived", ax=ax)
    st.pyplot(fig)

with col6:
    st.markdown("### Gender vs Class")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered_df, x="Sex", hue="Pclass", ax=ax)
    st.pyplot(fig)

# Row 3
col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("### Age vs Fare Scatter")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Survived", ax=ax)
    st.pyplot(fig)

with col8:
    st.markdown("### Boxplot: Age by Survival")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered_df, x="Survived", y="Age", ax=ax)
    st.pyplot(fig)

with col9:
    st.markdown("### Heatmap: Correlation Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

