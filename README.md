# Machine Learning Workshop Projects

This repository contains various machine learning projects and analyses focused on HR analytics and payroll modeling. The projects demonstrate practical applications of machine learning in business contexts.

## Projects

### 1. Employee Attrition and Performance Analysis
Located in: `/employee-attrition-and-performance/`

This project analyzes IBM HR Analytics Employee Attrition & Performance dataset to:
- Identify factors contributing to employee attrition
- Analyze correlations between various HR metrics
- Create predictive models for employee attrition
- Visualize key insights using seaborn and matplotlib

#### Running the Attrition Analysis

1. Navigate to the project directory:
```bash
cd employee-attrition-and-performance
```

2. Start Jupyter Notebook:
```bash
jupyter notebook Payroll-analysis.ipynb
```

3. In the notebook, you can:
- Run all cells to see the complete analysis
- Modify parameters to experiment with different models
- View visualizations of employee metrics
- Export results to various formats

### 2. Payroll Modeling System
Located in: `/payroll-modelling/`

A full-stack machine learning application for payroll prediction with:
- FastAPI backend for model serving
- Streamlit frontend for user interaction
- Docker support for easy deployment
- Cross-platform compatibility

#### Running the Payroll System

##### Option 1: Using Docker (Recommended)

1. Install Docker and Docker Compose:
   - [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - [Docker Engine for Linux](https://docs.docker.com/engine/install/)

2. Navigate to the payroll project:
```bash
cd payroll-modelling
```

3. Build and run services:
```bash
docker-compose up --build
```

4. Access the applications:
   - Web UI: http://localhost:8501
   - API Documentation: http://localhost:8000/docs

##### Option 2: Manual Installation

1. Create a virtual environment:
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
cd payroll-modelling
pip install -r requirements.txt
```

3. Train the model:
```bash
python payroll_train.py
```

4. Start the API server (in one terminal):
```bash
# Windows
uvicorn payroll_inference:app --host 0.0.0.0 --port 8000

# macOS/Linux
python -m uvicorn payroll_inference:app --host 0.0.0.0 --port 8000
```

5. Start the UI (in another terminal):
```bash
streamlit run payroll_ui.py
```

#### Using the Payroll System

1. Enter quarterly payment data in the UI
2. Click "Predict Annual Salary"
3. View the prediction and visualization
4. Use the API directly via:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"q1_payments": 50000.0, "q2_payments": 52000.0, "q3_payments": 51000.0, "q4_payments": 53000.0}'
```

## Technologies Used

### Common Dependencies
- Python 3.9+
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Jupyter Notebook

### Payroll System Additional Dependencies
- FastAPI
- Streamlit
- scikit-learn
- Docker (optional)
- uvicorn

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Git
- Docker (optional, but recommended)
- 4GB RAM minimum
- 2GB free disk space

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/ekkirinaldi/workshop-machine-learning.git
cd workshop-machine-learning
```

2. Install base dependencies:
```bash
pip install pandas numpy seaborn matplotlib jupyter
```

3. Install project-specific dependencies:
```bash
# For Payroll System
cd payroll-modelling
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. Port Conflicts
   - Error: "Address already in use"
   - Solution: Change ports in configuration or stop conflicting services

2. Memory Issues
   - Error: "Memory error" or slow performance
   - Solution: Close other applications or increase available memory

3. Python Version Conflicts
   - Error: "SyntaxError" or "ImportError"
   - Solution: Ensure you're using Python 3.9 or higher

### Platform-Specific Issues

#### Windows
- Use double backslashes or forward slashes in paths
- Run Command Prompt as Administrator if needed
- Use `python` instead of `python3`

#### macOS/Linux
- Ensure proper file permissions
- Use `python3` command explicitly
- Set execute permissions: `chmod +x *.sh`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

- GitHub: [@ekkirinaldi](https://github.com/ekkirinaldi)
- Project Link: [workshop-machine-learning](https://github.com/ekkirinaldi/workshop-machine-learning)