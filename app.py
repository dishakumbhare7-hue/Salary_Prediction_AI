import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="Salary Prediction AI",
    page_icon="💼",
    layout="wide"
)

# ----------------------------------
# Custom CSS
# ----------------------------------
st.markdown("""
<style>

/* All markdown text visible */
.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4,
.stMarkdown p {
    color: black !important;
    opacity: 1 !important;
}


/* White card text fix */
div[style*="background"] {
    background-color: white !important;
}

div[style*="background"] h1,
div[style*="background"] h2,
div[style*="background"] h3,
div[style*="background"] h4,
div[style*="background"] p {
    color: black !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: white !important;
    color: black !important;
}


/* Button */
.stButton > button:hover {
    background-color: #45a049 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------
# Load Model & Dataset
# ----------------------------------
model = joblib.load("salary_model.pkl")
data = pd.read_csv("dataset/salary_data.csv")

# Accuracy Calculation
X = data[["Experience","Skills"]]
y = data["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y_pred = model.predict(X_test)
accuracy = max(0, r2_score(y_test,y_pred)*100)
# ----------------------------------
# Sidebar
# ----------------------------------
st.sidebar.title("📌 Project Details")
st.sidebar.markdown("""
### Salary Prediction Using AI
**Algorithm**  
- Linear Regression  

**Programming Language**  
- Python  

**Libraries**  
- Streamlit  
- Pandas  
- Scikit-Learn  
- Joblib  
- Matplotlib  
""")
st.sidebar.success("AI Mini Project")

# ----------------------------------
# Heading
# ----------------------------------
st.title("💼 Salary Prediction Using AI")
st.write("Predict salary based on **Experience** and **Skill Score** using Machine Learning.")
st.divider()

# ----------------------------------
# Top Information Cards
# ----------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="background-color:white;padding:20px;border-radius:12px;
    border:1px solid #ddd;text-align:center;">
    <h4 style="color:#000000;">🤖 AI Model</h4>
   <h2 style="color:green !important;">
Linear Regression
</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background-color:white;padding:20px;border-radius:12px;
    border:1px solid #ddd;text-align:center;">
    <h4 style="color:#000000;">📂 Dataset</h4>
    <h2 style="color:green !important;">
11 Records
</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style="
    background:white;
    padding:20px;
    border-radius:12px;
    border:2px solid black;
    text-align:center;">

    <p style="color:black;font-size:20px;">
    🎯 Accuracy
    </p>

    <p style="color:red;font-size:30px;font-weight:bold;">
    {accuracy:.2f}%
    </p>

    </div>
    """, unsafe_allow_html=True)
# ----------------------------------
# User Input
# ----------------------------------
experience = st.number_input("👨‍💻 Years of Experience",min_value=0,max_value=50,value=1)
skills = st.slider("⭐ Skill Score",1,20,5)

# ----------------------------------
# Prediction
# ----------------------------------
if st.button("🔍 Predict Salary"):
    input_data = pd.DataFrame({"Experience": [experience], "Skills": [skills]})
    prediction = model.predict(input_data)[0]
    
    # ✅ History append fix
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({
        "Experience": experience,
        "Skill Score": skills,
        "Predicted Salary": round(prediction)
    })

    st.success("✅ Prediction Completed Successfully!")
    
    st.markdown(f"""
<div style="
background:white;
padding:20px;
border-radius:12px;
border:1px solid #ddd;
box-shadow:0 2px 8px rgba(0,0,0,0.2);
margin-bottom:20px;
text-align:center;">

<h3 style="color:black !important;">
💰 Estimated Salary
</h3>

<h1 style="color:#28a745 !important; font-size:40px; font-weight:bold;">
₹ {prediction:,.0f}
</h1>

</div>
""", unsafe_allow_html=True)

    # Salary Category
    if prediction < 400000:
        st.warning("🟡 Fresher Level Salary")
    elif prediction < 800000:
        st.info("🔵 Intermediate Level Salary")
    else:
        st.success("🟢 Expert Level Salary")

    # Prediction Details
    st.subheader("📋 Prediction Details")

    c1, c2, c3 = st.columns(3)

    c1.metric("Experience", f"{experience} Years")
    c2.metric("Skill Score", f"{skills}")
    c3.metric("Predicted Salary", f"₹ {prediction:,.0f}")
# ----------------------------------
# Dataset Section
# ----------------------------------

st.divider()

st.subheader("📄 Dataset Section")

show_dataset = st.checkbox("Show Dataset", key="dataset_new")

if show_dataset:

    st.dataframe(data, use_container_width=True)

    st.write("Dataset Loaded Successfully")

    st.subheader("📊 Salary Statistics")

    avg = data["Salary"].mean()
    high = data["Salary"].max()
    low = data["Salary"].min()

    st.write("Average:", avg)
    st.write("Highest:", high)
    st.write("Lowest:", low)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Average Salary",
            value=f"₹ {int(avg):,}"
        )

    with col2:
        st.metric(
            label="Highest Salary",
            value=f"₹ {int(high):,}"
        )

    with col3:
        st.metric(
            label="Lowest Salary",
            value=f"₹ {int(low):,}"
        )


    st.subheader("📈 Experience vs Salary")

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(
        data["Experience"],
        data["Salary"],
        marker="o",
        linewidth=2,
        color="green"
    )

    ax.set_xlabel("Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Experience vs Salary")

    st.pyplot(fig)


    st.subheader("📊 Salary Comparison")

    fig2, ax2 = plt.subplots(figsize=(8,4))

    ax2.bar(
        data["Experience"],
        data["Salary"],
        color="orange"
    )

    ax2.set_xlabel("Experience")
    ax2.set_ylabel("Salary")
    ax2.set_title("Salary Comparison")

    st.pyplot(fig2)


    csv = data.to_csv(index=False)

    st.download_button(
        "📥 Download Dataset",
        csv,
        file_name="salary_data.csv",
        mime="text/csv"
    )


# ----------------------------------
# Prediction History
# ----------------------------------

st.divider()

st.subheader("📜 Prediction History")

if "history" in st.session_state and len(st.session_state.history)>0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction History",
        csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:
    st.info("No predictions yet.")