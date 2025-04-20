import streamlit as st
import requests
import os
from urllib.parse import urljoin
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_api_url():
    """Get the API URL from environment variable or use default."""
    api_url = os.getenv("API_URL", "http://localhost:8000")
    return api_url.rstrip('/')  # Remove trailing slash if present

def check_api_health(api_url: str) -> bool:
    """Check if the API is healthy and responding."""
    try:
        response = requests.get(api_url, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed: {str(e)}")
        return False

def create_streamlit_ui():
    """Create and run the Streamlit user interface."""
    try:
        st.set_page_config(
            page_title="Employee Attrition Prediction",
            page_icon="👥",
            layout="wide"
        )
        
        st.title("🏢 Employee Attrition Prediction")
        st.write("This application predicts the likelihood of an employee leaving the company.")
        
        # Get API URL and check health
        api_url = get_api_url()
        
        # System information
        with st.expander("System Information"):
            st.write(f"Operating System: {platform.system()} {platform.release()}")
            st.write(f"API URL: {api_url}")
            if check_api_health(api_url):
                st.success("✅ API is running and healthy")
            else:
                st.error("❌ API is not available")
                st.info("Please make sure the inference service is running")
                return
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📋 Personal Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            education = st.number_input("Education Level (1-5)", min_value=1, max_value=5, value=3)
            education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other", "Human Resources"])
            
            st.header("💼 Job Information")
            department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
            job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
            job_level = st.number_input("Job Level (1-5)", min_value=1, max_value=5, value=2)
            years_at_company = st.number_input("Years at Company", min_value=0, value=5)
            
        with col2:
            st.header("📊 Performance Metrics")
            job_satisfaction = st.number_input("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
            environment_satisfaction = st.number_input("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
            work_life_balance = st.number_input("Work Life Balance (1-4)", min_value=1, max_value=4, value=3)
            relationship_satisfaction = st.number_input("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
            
            st.header("💰 Compensation")
            monthly_income = st.number_input("Monthly Income", min_value=0, value=5000)
            percent_salary_hike = st.number_input("Percent Salary Hike", min_value=0, value=15)
            stock_option_level = st.number_input("Stock Option Level (0-3)", min_value=0, max_value=3, value=1)
        
        # Additional inputs with default values
        input_data = {
            "Age": age,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 1000,
            "Department": department,
            "DistanceFromHome": 10,
            "Education": education,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": 50,
            "JobInvolvement": 3,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MaritalStatus": marital_status,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": 15000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": 8,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 3
        }
        
        if st.button("🔍 Predict Attrition", type="primary"):
            try:
                # Make prediction request
                response = requests.post(
                    urljoin(api_url, "predict"),
                    json=input_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.header("🎯 Prediction Results")
                    
                    # Create a container for the results
                    results_container = st.container()
                    
                    with results_container:
                        prob = result["probability_of_attrition"]
                        prediction = result["prediction"]
                        confidence = result.get("confidence", prob)
                        
                        # Display the probability with a progress bar
                        st.write("Probability of Attrition:")
                        st.progress(prob)
                        st.write(f"{prob:.1%}")
                        
                        # Display the confidence
                        st.write("Confidence Score:")
                        st.info(f"{confidence:.1%}")
                        
                        # Display the prediction with colored box
                        if prediction == "Yes":
                            st.error(f"⚠️ Prediction: {prediction} - High Risk of Attrition")
                        else:
                            st.success(f"✅ Prediction: {prediction} - Low Risk of Attrition")
                else:
                    error_msg = response.json().get('detail', 'Unknown error occurred')
                    st.error(f"❌ Error: {error_msg}")
                    logger.error(f"API error: {error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to the API. Please make sure the inference service is running.")
                logger.error("Connection error when trying to reach the API")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Please try again.")
                logger.error("Request to API timed out")
            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}")
                st.error("Please make sure the inference service is running")

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ Application error: {str(e)}")

if __name__ == "__main__":
    create_streamlit_ui()
