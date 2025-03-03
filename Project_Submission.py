import streamlit as st
import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import MinMaxScaler

# Title
st.title("📚 AI-Powered Group Project Submission & Fair Evaluation")

# Sidebar for user role selection
role = st.sidebar.selectbox("Select Role", ["Student", "Faculty"])

# Define evaluation criteria
criteria = ["Contribution Score", "Code Quality", "Documentation", "Teamwork"]

if role == "Student":
    st.header("📝 Submit Your Group Project")
    student_name = st.text_input("Enter Your Name")
    group_id = st.text_input("Enter Group ID")
    project_file = st.file_uploader("Upload Project (ZIP/PDF)", type=["zip", "pdf"])
    
    # Collect self-evaluation scores
    st.subheader("Self-Evaluation")
    student_scores = {criterion: st.slider(criterion, 0, 10, 5) for criterion in criteria}
    
    if st.button("Submit Project"):
        st.success(f"✅ Project Submitted Successfully by {student_name} in Group {group_id}!")

elif role == "Faculty":
    st.header("📊 Evaluate Group Projects Fairly")
    uploaded_file = st.file_uploader("Upload Project Scores (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📂 Uploaded Data:", df.head())

        # Normalize Scores
        scaler = MinMaxScaler()
        df[criteria] = scaler.fit_transform(df[criteria])
        
        # Convert dataset to AIF360 format
        dataset = BinaryLabelDataset(
            df=df,
            label_names=["Final Score"],
            protected_attribute_names=["Gender"]  # Ensure fairness across genders
        )
        
        # Compute Fairness Metrics
        metric = BinaryLabelDatasetMetric(
            dataset, privileged_groups=[{"Gender": "Male"}], unprivileged_groups=[{"Gender": "Female"}]
        )
        
        st.subheader("Fairness Evaluation")
        st.write("📊 Statistical Parity Difference:", metric.statistical_parity_difference())
        st.write("⚖️ Disparate Impact:", metric.disparate_impact())
        
        if st.button("Finalize Evaluations"):
            st.success("✅ Evaluations Finalized & Ready for Integration with Samarth!")