# Payroll Prediction System

A machine learning-based system for predicting annual salaries based on quarterly payment data. The system consists of a FastAPI backend for predictions and a Streamlit frontend for user interaction.

## Table of Contents
- [Features](#features)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
  - [Using Docker (Recommended)](#using-docker-recommended)
  - [Manual Installation](#manual-installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Features
- Machine learning model for salary prediction
- RESTful API for model inference
- User-friendly web interface
- Real-time predictions
- Data visualization
- Cross-platform compatibility (Windows, macOS, Linux)

## System Requirements
- Python 3.11 or higher
- Docker (optional, but recommended)
- 4GB RAM minimum
- 2GB free disk space

## Project Structure
```
payroll-modelling/
├── data-payroll.csv          # Training data
├── payroll_train.py          # Model training script
├── payroll_inference.py      # FastAPI server
├── payroll_ui.py            # Streamlit UI
├── requirements.txt         # Python dependencies
├── Dockerfile.api          # API service Dockerfile
├── Dockerfile.ui          # UI service Dockerfile
└── docker-compose.yml    # Docker services configuration
```

## Installation

### Using Docker (Recommended)

1. Install Docker and Docker Compose:
   - Windows: Install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - macOS: Install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - Linux: Install [Docker Engine](https://docs.docker.com/engine/install/) and [Docker Compose](https://docs.docker.com/compose/install/)

2. Clone or download this repository:
   ```bash
   git clone <repository-url>
   cd payroll-modelling
   ```

3. Build and run the services:
   ```bash
   docker-compose up --build
   ```

### Manual Installation

1. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python payroll_train.py
   ```

4. Start the API server:
   ```bash
   # Windows
   uvicorn payroll_inference:app --host 0.0.0.0 --port 8000

   # macOS/Linux
   python -m uvicorn payroll_inference:app --host 0.0.0.0 --port 8000
   ```

5. Start the UI (in a new terminal):
   ```bash
   streamlit run payroll_ui.py
   ```

## Usage

1. Access the web interface:
   - Open your browser and go to http://localhost:8501

2. Enter quarterly payment data:
   - Input payment amounts for Q1, Q2, Q3, and Q4
   - Click "Predict Annual Salary"
   - View the prediction and visualization

3. API endpoints:
   - Prediction API: http://localhost:8000/predict
   - API documentation: http://localhost:8000/docs

## API Documentation

### Predict Endpoint

- URL: `/predict`
- Method: POST
- Request Body:
  ```json
  {
    "q1_payments": float,
    "q2_payments": float,
    "q3_payments": float,
    "q4_payments": float
  }
  ```
- Response:
  ```json
  {
    "projected_annual_salary": float
  }
  ```

### Health Check

- URL: `/health`
- Method: GET
- Response:
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

## Troubleshooting

### Common Issues

1. Port Conflicts
   - Error: "Address already in use"
   - Solution: Change ports in docker-compose.yml or stop conflicting services

2. Model Loading Error
   - Error: "Model file not found"
   - Solution: Run training script first: `python payroll_train.py`

3. Memory Issues
   - Error: "Memory error" or slow performance
   - Solution: Close other applications or increase Docker memory limit

4. Connection Error
   - Error: "Cannot connect to API"
   - Solution: Check if API service is running and accessible

### Platform-Specific Issues

#### Windows
- If you get path-related errors, use double backslashes or forward slashes in paths
- Run commands in Command Prompt as Administrator if needed

#### macOS/Linux
- Ensure proper permissions for files and directories
- Use `python3` instead of `python` if needed

## Development

### Running Tests
```bash
# Unit tests
python -m pytest tests/

# Integration tests
python -m pytest tests/integration/
```

### Code Style
```bash
# Format code
black .

# Lint code
flake8
```

### Environment Variables
- `API_URL`: API server URL (default: http://localhost:8000)
- `PORT`: API server port (default: 8000)
- `HOST`: API server host (default: 0.0.0.0)

### Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
