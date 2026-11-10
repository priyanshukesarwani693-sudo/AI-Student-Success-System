import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

from pathlib import Path
from io import BytesIO
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(
    page_title="AI Student Success System",
    page_icon="🎓",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_model.pkl"
DATA_PATH = BASE_DIR / "data" / "student_data.csv"

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip().str.lower()
    return data

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)

st.sidebar.title("🎓 Student AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "AI Prediction",
        "Career Recommendation",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("AI Student Success System")

st.title("AI Student Success & Career Prediction")
st.caption("Intelligent Academic Performance Analytics Platform")


if page == "Dashboard":

    st.header("Dashboard Overview")

    try:
        data = load_data()

        c1, c2, c3 = st.columns(3)

        c1.metric("Total Students", len(data))

        c2.metric(
            "Average Marks",
            f"{data['final_marks'].mean():.1f}%"
        )

        c3.metric(
            "Average Attendence",
            f"{data['attendence'].mean():.1f}%"
        )

        st.subheader("Student Performance Analysis")

        fig = px.bar(
            data,
            x="student_id",
            y="final_marks",
            title="Student-wise Final Marks",
            labels={
                "student_id": "Student ID",
                "final_marks": "Final Marks"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.scatter(
            data,
            x="attendence",
            y="final_marks",
            color="study_hours",
            title="Attendence vs Final Marks",
            labels={
                "attendence": "Attendence",
                "final_marks": "Final Marks",
                "study_hours": "Study Hours"
            }
        )

        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Student Dataset")
       
        st.text(data.to_string(index=False))

    except Exception as e:
        st.error(f"Dashboard Error: {e}")


elif page == "AI Prediction":

    st.header("AI Student Performance Prediction")

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input("Student Name")

        attendence = st.slider(
            "Attendence (%)",
            0, 100, 75
        )

        study_hours = st.number_input(
            "Daily Study Hours",
            min_value=0.0,
            max_value=24.0,
            value=3.0
        )

    with col2:

        previous_marks = st.number_input(
            "Previous Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0
        )

        assignment_score = st.number_input(
            "Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=70.0
        )

    if st.button("Predict Performance", type="primary"):

        if not name.strip():

            st.warning("Please enter student name.")

        else:

            try:

                model = load_model()

                input_data = pd.DataFrame([{
                    "attendence": attendence,
                    "study_hours": study_hours,
                    "previous_marks": previous_marks,
                    "assignment_score": assignment_score
                }])

                prediction = model.predict(input_data)[0]
                prediction = max(0, min(100, prediction))

                st.success("Prediction generated successfully!")

                st.metric(
                    "Predicted Final Marks",
                    f"{prediction:.2f}/100"
                )

                st.progress(int(prediction))

                if prediction >= 75:
                    st.success("Performance: Excellent")

                elif prediction >= 50:
                    st.info("Performance: Average")

                else:
                    st.warning("Performance: Needs Improvement")

                st.caption(
                    "Prediction based on your trained Random Forest model."
                )

                pdf_buffer = BytesIO()

                pdf = canvas.Canvas(pdf_buffer)

                pdf.setTitle("Student Performance Report")

                pdf.setFont("Helvetica-Bold", 18)
                pdf.drawString(
                    50, 800,
                    "Student Performance Report"
                )

                pdf.setFont("Helvetica", 11)

                pdf.drawString(
                    50, 760,
                    f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}"
                )

                pdf.drawString(
                    50, 730,
                    f"Student Name: {name}"
                )

                pdf.drawString(
                    50, 700,
                    f"Attendence: {attendence}%"
                )

                pdf.drawString(
                    50, 675,
                    f"Study Hours: {study_hours}"
                )

                pdf.drawString(
                    50, 650,
                    f"Previous Marks: {previous_marks}"
                )

                pdf.drawString(
                    50, 625,
                    f"Assignment Score: {assignment_score}"
                )

                pdf.setFont("Helvetica-Bold", 14)

                pdf.drawString(
                    50, 580,
                    f"Predicted Final Marks: {prediction:.2f}/100"
                )

                pdf.setFont("Helvetica", 10)

                pdf.drawString(
                    50, 540,
                    "Generated using Random Forest Regression."
                )

                pdf.drawString(
                    50, 520,
                    "This is an estimated prediction, not actual exam marks."
                )

                pdf.save()

                pdf_buffer.seek(0)

                st.download_button(
                    label="Download Student Report (PDF)",
                    data=pdf_buffer,
                    file_name="Student_Performance_Report.pdf",
                    mime="application/pdf"
                )

            except Exception as e:

                st.error(f"Prediction Error: {e}")


elif page == "Career Recommendation":

    st.header("Career Recommendation System")

    interest = st.selectbox(
        "Select Your Interest",
        [
            "Data Science",
            "Artificial Intelligence",
            "Web Development",
            "Cyber Security",
            "Software Development"
        ]
    )

    skills = st.multiselect(
        "Select Your Skills",
        [
            "Python",
            "C++",
            "SQL",
            "Machine Learning",
            "HTML",
            "CSS",
            "JavaScript",
            "Statistics"
        ]
    )

    if st.button("Get Career Recommendation"):

        career_map = {
            "Data Science": [
                "Data Analyst",
                "Data Scientist",
                "Business Intelligence Analyst"
            ],

            "Artificial Intelligence": [
                "AI Engineer",
                "Machine Learning Engineer",
                "AI Research Assistant"
            ],

            "Web Development": [
                "Frontend Developer",
                "Backend Developer",
                "Full Stack Developer"
            ],

            "Cyber Security": [
                "Security Analyst",
                "Penetration Tester",
                "Security Engineer"
            ],

            "Software Development": [
                "Software Engineer",
                "Python Developer",
                "Application Developer"
            ]
        }

        st.success("Recommended Career Paths")

        for career in career_map[interest]:
            st.write("✅", career)

        st.write(
            "Your selected skills:",
            ", ".join(skills) if skills else "No skills selected"
        )

        st.caption(
            "These are interest-based suggestions, not ML-generated predictions."
        )


elif page == "About Project":

    st.header("About This Project")

    st.write(
        "AI Student Success System is a Machine Learning "
        "based application designed to analyze academic "
        "performance and predict student outcomes."
    )

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Random Forest Regression
    - Streamlit
    - Plotly
    - ReportLab
    - Machine Learning
    """)

    st.info(
        "This project is developed for academic learning "
        "and student performance analysis."
    )