# Employee Attrition Prediction System

This project implements a machine learning system for predicting employee attrition using XGBoost. The system consists of three main components:
1. Model Training (`model_train.py`)
2. Inference API (`inference.py`)
3. Web Interface (`main.py`)

## Project Structure

```
employee-attrition-and-performance/
├── data-employee-attrition.csv    # Dataset file
├── model_train.py                 # Model training script
├── inference.py                   # FastAPI inference service
├── main.py                        # Streamlit web interface
├── requirements.txt               # Python dependencies
├── Dockerfile.inference           # Dockerfile for inference service
├── Dockerfile.web                # Dockerfile for web interface
└── docker-compose.yml            # Docker Compose configuration
```

## Features

- **Model Training**:
  - XGBoost classifier for attrition prediction
  - Automated data preprocessing
  - Model evaluation with accuracy metrics
  - Cross-platform compatible model persistence
  - Detailed progress reporting

- **Inference API**:
  - RESTful API using FastAPI
  - Real-time predictions
  - Input validation
  - Comprehensive error handling
  - Health check endpoint
  - API documentation (Swagger UI)
  - Cross-platform path handling

- **Web Interface**:
  - User-friendly interface using Streamlit
  - Intuitive form inputs
  - Real-time predictions
  - Visual presentation of results
  - System information display
  - API health monitoring
  - Error handling and user feedback
  - Cross-platform compatibility

## Requirements

- Python 3.11+
- Docker (optional, for containerized deployment)
- Operating System: Windows, macOS, or Linux

## Installation & Setup

### Option 1: Local Development

1. Clone the repository and navigate to the project directory

2. Create a virtual environment:
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Deployment

Ensure Docker and Docker Compose are installed on your system:
- [Docker Installation Guide](https://docs.docker.com/get-docker/)
- Docker Compose is included with Docker Desktop for Windows and macOS

## Running the Application

### Method 1: Local Development (Multiple Terminals)

1. Train the model (Terminal 1):
```bash
python model_train.py
```

2. Start the inference API (Terminal 1 or new terminal):
```bash
python inference.py
```

3. Launch the web interface (New terminal):
```bash
streamlit run main.py
```

### Method 2: Docker Deployment (Single Command)

```bash
docker compose up --build
```

## Accessing the Services

### Local Development
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:8000/docs
- API Health Check: http://localhost:8000

### Docker Deployment
Same URLs as local development, but services run in containers.

## Component Details

### 1. Model Training (`model_train.py`)

Features:
- Cross-platform path handling
- Progress reporting
- Error handling
- Model evaluation metrics

```bash
python model_train.py
```

Output:
- Trained model file: `xgboost_model.pkl`
- Performance metrics display
- Classification report

### 2. Inference API (`inference.py`)

Endpoints:
- GET `/`: Health check and API information
- POST `/predict`: Make predictions
- GET `/docs`: API documentation

Features:
- Automatic model loading
- Input validation
- Error handling
- Cross-platform compatibility

### 3. Web Interface (`main.py`)

Features:
- System information display
- API health monitoring
- Real-time predictions
- Error handling
- Cross-platform compatibility

## Troubleshooting

### Common Issues

1. **Port Conflicts**
```bash
# Check if ports are in use
# Windows (PowerShell)
netstat -ano | findstr "8000 8501"

# macOS/Linux
lsof -i :8000,8501
```

2. **API Connection Issues**
- Verify the inference service is running
- Check the API_URL environment variable
- Ensure no firewall blocking

3. **Model Loading Issues**
- Confirm model training was successful
- Check file permissions
- Verify path compatibility

4. **Docker Issues**
```bash
# Remove old containers
docker compose down

# Clean up images
docker system prune

# Rebuild
docker compose up --build
```

### Platform-Specific Notes

#### Windows
- Use backslashes or raw strings for paths
- Run commands in PowerShell or Command Prompt
- Check Windows Defender Firewall

#### macOS/Linux
- Use forward slashes for paths
- Check file permissions
- Verify Python installation

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| PORT | Inference API port | 8000 |
| API_URL | API endpoint URL | http://localhost:8000 |

## Logging

- Application logs are available in the console
- Docker logs can be viewed with:
```bash
docker compose logs -f
```

## Security Notes

- API runs on localhost by default
- No authentication required (development setup)
- Model file should have appropriate permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Testing

Run the application on different platforms to ensure compatibility:
1. Windows 10/11
2. macOS
3. Linux (Ubuntu/Debian)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- XGBoost team for the machine learning library
- FastAPI team for the API framework
- Streamlit team for the web interface framework 