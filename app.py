import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")

# Background Image from GitHub
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

# Title
st.markdown("""
<div style="background-color: rgba(255,255,255,0.8); padding: 10px; border-radius: 10px;">
    <h1 style="text-align: center;">🚢 Titanic Data Analytics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Load Data
df = pd.read_csv("cleaned_titanic.csv")

# Sidebar Filters
st.sidebar.header("Filter Options")
gender = st.sidebar.selectbox("Select Gender", options=df["Sex"].unique())
pclass = st.sidebar.selectbox("Select Passenger Class", options=sorted(df["Pclass"].unique()))
embarked = st.sidebar.multiselect("Select Embarked", options=df["Embarked"].dropna().unique(), default=df["Embarked"].dropna().unique())
survived = st.sidebar.selectbox("Survival Status", options=[0, 1, "Both"])

age_range = st.sidebar.slider("Select Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))

# Apply Filters
filtered_df = df.copy()
filtered_df = filtered_df[
    (filtered_df["Sex"] == gender) &
    (filtered_df["Pclass"] == pclass) &
    (filtered_df["Embarked"].isin(embarked)) &
    (filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1])
]

if survived != "Both":
    filtered_df = filtered_df[filtered_df["Survived"] == survived]

# Show Filtered Data
if st.checkbox("Show Raw Data"):
    st.dataframe(filtered_df)

# Grid Layout 3x3 Visualizations
st.markdown("### 📊 Visual Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Survival Count by Gender")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x="Survived", hue="Sex", ax=ax1)
    st.pyplot(fig1)

with col2:
    st.markdown("#### Age Distribution")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df["Age"], kde=True, ax=ax2, bins=20)
    st.pyplot(fig2)

with col3:
    st.markdown("#### Fare Distribution by Class")
    fig3, ax3 = plt.subplots()
    sns.boxplot(data=filtered_df, x="Pclass", y="Fare", ax=ax3)
    st.pyplot(fig3)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("#### Survival by Class")
    fig4, ax4 = plt.subplots()
    sns.countplot(data=filtered_df, x="Pclass", hue="Survived", ax=ax4)
    st.pyplot(fig4)

with col5:
    st.markdown("#### Gender Distribution")
    fig5, ax5 = plt.subplots()
    sns.countplot(data=filtered_df, x="Sex", ax=ax5)
    st.pyplot(fig5)

with col6:
    st.markdown("#### Embarked Distribution")
    fig6, ax6 = plt.subplots()
    sns.countplot(data=filtered_df, x="Embarked", ax=ax6)
    st.pyplot(fig6)

col7, col8, col9 = st.columns(3)

with col7:
    st.markdown("#### Age vs Fare")
    fig7, ax7 = plt.subplots()
    sns.scatterplot(data=filtered_df, x="Age", y="Fare", hue="Survived", ax=ax7)
    st.pyplot(fig7)

with col8:
    st.markdown("#### Average Fare by Embarkment")
    fig8, ax8 = plt.subplots()
    sns.barplot(data=filtered_df, x="Embarked", y="Fare", estimator='mean', ax=ax8)
    st.pyplot(fig8)

with col9:
    st.markdown("#### Class vs Survival Heatmap")
    heatmap_data = pd.crosstab(filtered_df['Pclass'], filtered_df['Survived'])
    fig9, ax9 = plt.subplots()
    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlGnBu", ax=ax9)
    st.pyplot(fig9)
