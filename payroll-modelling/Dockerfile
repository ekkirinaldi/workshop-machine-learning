FROM python:3.9-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY payroll_train.py .
COPY payroll_inference.py .
COPY payroll_ui.py .

# Train the model during build
COPY data-payroll.csv .
RUN python payroll_train.py

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501 