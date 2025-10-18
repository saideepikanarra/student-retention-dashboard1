import os
import pandas as pd
import requests
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

# ======================================
# Clear previous Streamlit session (prevents old widgets showing)
# ======================================
st.session_state.clear()  

# ======================================
# 🔑 API Keys
# ======================================
# ✅ Load from Streamlit Secrets (secure)
os.environ["CREWAI_BEARER_TOKEN"] = st.secrets["CREWAI_BEARER_TOKEN"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# CrewAI API
BASE_URL = "https://enrollment-retention-management-crew-v1-c21-82c3d780.crewai.com"
KICKOFF_URL = f"{BASE_URL}/kickoff"
STATUS_URL = f"{BASE_URL}/status"
HEADERS = {
    "Authorization": f"Bearer {os.environ['CREWAI_BEARER_TOKEN']}",
    "Content-Type": "application/json"
}

# ======================================
# Initialize Groq LLM
# ======================================
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

# ======================================
# Department Prompt Template
# ======================================
dept_template = """
You are an academic management strategist.
Summarize the department-level performance from an administration perspective.

Department: {Department}
Average GPA: {Avg_GPA}
Average Attendance: {Avg_Attendance}%
Average Engagement: {Avg_Engagement}

Highlight key strengths, risks, and actionable recommendations for retention and enrollment.
"""

dept_prompt = PromptTemplate(
    template=dept_template,
    input_variables=["Department", "Avg_GPA", "Avg_Attendance", "Avg_Engagement"]
)

dept_chain = LLMChain(llm=llm, prompt=dept_prompt)

# ======================================
# CrewAI Functions
# ======================================
def start_crew(department_name):
    data = {"inputs": {"department_name": department_name}}
    response = requests.post(KICKOFF_URL, json=data, headers=HEADERS)
    response.raise_for_status()
    return response.json()["kickoff_id"]

def check_status(kickoff_id):
    import time
    while True:
        response = requests.get(f"{STATUS_URL}/{kickoff_id}", headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        if result.get("state") == "SUCCESS":
            return result.get("result", "No output returned")
        elif result.get("state") == "FAILED":
            return "❌ Crew execution failed"
        time.sleep(2)

# ======================================
# Streamlit UI
# ======================================
st.title("🏫 Student Enrollment & Retention Strategy")
st.write("Upload a department CSV, select a department, and generate insights via LangChain + CrewAI.")

uploaded_file = st.file_uploader("Upload Department CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

    # Convert categorical Engagement to numeric to prevent errors
    engagement_map = {"Low": 1, "Medium": 2, "High": 3}
    df["Engagement_numeric"] = df["Engagement"].map(engagement_map).fillna(0)

    if "Department" in df.columns:
        department_name = st.selectbox("Select a Department", df["Department"].unique())

        if st.button("Analyze Department"):
            dept_df = df[df["Department"] == department_name]
            avg_gpa = round(dept_df["GPA"].mean(), 2)
            avg_attendance = round(dept_df["Attendance"].mean(), 2)
            avg_engagement = round(dept_df["Engagement_numeric"].mean(), 2)

            # LangChain summary
            dept_summary = dept_chain.run({
                "Department": department_name,
                "Avg_GPA": avg_gpa,
                "Avg_Attendance": avg_attendance,
                "Avg_Engagement": avg_engagement
            })
            st.subheader("📊 Department Summary (LangChain via Groq)")
            st.write(dept_summary)

            # CrewAI retention strategy
            try:
                kickoff_id = start_crew(department_name)
                analysis = check_status(kickoff_id)
                st.subheader("🤖 CrewAI Department Retention Strategy")
                st.write(analysis)
            except Exception as e:
                st.error(f"CrewAI request failed: {e}")

else:
    st.info("Upload a CSV file to begin analysis.")
