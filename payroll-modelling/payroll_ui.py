import os
import time
import pandas as pd
import numpy as np
import requests
import streamlit as st

# Set page config first before any other Streamlit commands
st.set_page_config(
    page_title="Payroll Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'api_healthy' not in st.session_state:
    st.session_state.api_healthy = False

# Get API URL from environment variable or use default
API_URL = os.getenv('API_URL', 'http://localhost:8000')

def check_api_health():
    """Check if the API is healthy"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def format_currency(amount):
    """Format amount as currency"""
    return f"${amount:,.2f}"

def display_quarterly_chart(payments_data):
    """Display a bar chart of quarterly payments using Streamlit"""
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    df = pd.DataFrame({
        "Quarter": quarters,
        "Payment": payments_data
    })
    
    # Display bar chart using Streamlit
    st.bar_chart(df.set_index("Quarter"))
    
    # Display table with formatted values
    st.write("Quarterly Payments Details:")
    display_df = pd.DataFrame({
        "Quarter": quarters,
        "Payment": [format_currency(p) for p in payments_data]
    })
    st.table(display_df)

def main():
    try:
        st.title("💰 Payroll Annual Salary Prediction")
        st.write("Enter quarterly payment information to predict annual salary")

        # Check API health
        if not st.session_state.api_healthy:
            with st.spinner("Checking API connection..."):
                st.session_state.api_healthy = check_api_health()
                time.sleep(1)  # Give time for API to start
                
        if not st.session_state.api_healthy:
            st.error(f"⚠️ Cannot connect to API at {API_URL}. Please check if the API service is running.")
            if st.button("Retry Connection"):
                st.session_state.api_healthy = check_api_health()
                st.experimental_rerun()
            return

        # Create input form
        with st.form("prediction_form"):
            st.write("### Enter Quarterly Payments")
            
            col1, col2 = st.columns(2)
            
            with col1:
                q1_payments = st.number_input(
                    "Q1 Payments ($)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )
                q2_payments = st.number_input(
                    "Q2 Payments ($)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )
            
            with col2:
                q3_payments = st.number_input(
                    "Q3 Payments ($)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )
                q4_payments = st.number_input(
                    "Q4 Payments ($)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=0.0,
                    step=1000.0,
                    format="%.2f"
                )

            submitted = st.form_submit_button("Predict Annual Salary")

        # Make prediction when form is submitted
        if submitted:
            # Validate inputs
            total_payments = sum([q1_payments, q2_payments, q3_payments, q4_payments])
            if total_payments == 0:
                st.warning("Please enter at least one quarterly payment amount.")
                return

            # Prepare data for API request
            data = {
                "q1_payments": float(q1_payments),
                "q2_payments": float(q2_payments),
                "q3_payments": float(q3_payments),
                "q4_payments": float(q4_payments)
            }

            # Show prediction section
            with st.container():
                try:
                    with st.spinner("Calculating prediction..."):
                        # Make API request with timeout
                        response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
                        response.raise_for_status()
                        
                        # Get prediction
                        prediction = float(response.json()["projected_annual_salary"])
                        
                        # Display results
                        st.markdown("### Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(
                                label="Predicted Annual Salary",
                                value=format_currency(prediction),
                                delta=format_currency(prediction - total_payments)
                            )
                        
                        with col2:
                            st.metric(
                                label="Total Quarterly Payments",
                                value=format_currency(total_payments)
                            )

                        # Display visualization
                        st.markdown("### Quarterly Payment Distribution")
                        payments = [q1_payments, q2_payments, q3_payments, q4_payments]
                        display_quarterly_chart(payments)

                except requests.exceptions.RequestException as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error(f"Make sure the API server is running at {API_URL}")
                    if st.button("Retry Request"):
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
                    st.error("Please try again or contact support if the error persists.")
                    if st.button("Retry"):
                        st.experimental_rerun()

    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please refresh the page or contact support if the error persists.")
        if st.button("Refresh Application"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
