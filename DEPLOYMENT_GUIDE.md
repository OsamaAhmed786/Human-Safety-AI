# Human Safety AI - Deployment Guide

## Overview

This deployment guide provides comprehensive instructions for deploying the Human Safety AI project in various environments, from local development to production systems. It covers setup procedures, configuration options, troubleshooting, and best practices for maintaining a robust AI-powered communication analysis system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Configuration Options](#configuration-options)
5. [Performance Optimization](#performance-optimization)
6. [Security Considerations](#security-considerations)
7. [Monitoring and Maintenance](#monitoring-and-maintenance)
8. [Troubleshooting](#troubleshooting)
9. [Scaling Considerations](#scaling-considerations)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.14+), or Windows 10
- **Python**: Version 3.8 or higher (3.11 recommended)
- **RAM**: 4GB minimum (8GB recommended for smooth operation)
- **Storage**: 2GB free space for models and dependencies
- **Network**: Internet connection for initial model downloads

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+ or CentOS 8+)
- **Python**: Version 3.11
- **RAM**: 16GB for optimal performance with multiple concurrent users
- **Storage**: 10GB free space for models, logs, and temporary files
- **CPU**: Multi-core processor (4+ cores recommended)
- **Network**: Stable broadband connection

### Hardware Considerations

**CPU Requirements**: The system is primarily CPU-bound, especially during text processing and model inference. Multi-core processors will improve performance when handling multiple requests.

**Memory Usage**: Transformer models require significant RAM. The emotion detection model alone uses approximately 1.3GB of memory when loaded. Plan for at least 2GB of available RAM per concurrent user.

**Storage Needs**: Pre-trained models are downloaded automatically on first use and cached locally. Ensure sufficient storage space for:
- Hugging Face models: ~1.5GB
- NLTK data: ~100MB
- Application code and dependencies: ~500MB
- Logs and temporary files: Variable

## Local Development Setup

### Step 1: Environment Preparation

Create a clean Python environment to avoid dependency conflicts:

```bash
# Using venv (recommended)
python3 -m venv human_safety_ai_env
source human_safety_ai_env/bin/activate  # On Windows: human_safety_ai_env\Scripts\activate

# Or using conda
conda create -n human_safety_ai python=3.11
conda activate human_safety_ai
```

### Step 2: Clone and Setup Project

```bash
# Clone the project (if using version control)
git clone <repository-url>
cd human_safety_ai

# Or create directory structure if starting from files
mkdir human_safety_ai
cd human_safety_ai
# Copy project files here
```

### Step 3: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, nltk, transformers; print('Dependencies installed successfully')"
```

### Step 4: Download Required Data

```bash
# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Test model loading (this will download the emotion model)
python -c "from src.emotion_analysis import EmotionAnalyzer; EmotionAnalyzer()"
```

### Step 5: Run the Application

```bash
# Start the Streamlit application
streamlit run src/app.py

# The application will be available at http://localhost:8501
```

### Development Environment Configuration

Create a `.streamlit/config.toml` file for development settings:

```toml
[server]
port = 8501
headless = false
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[logger]
level = "debug"
```

## Production Deployment

### Option 1: Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run the Docker container:

```bash
# Build the image
docker build -t human-safety-ai .

# Run the container
docker run -p 8501:8501 human-safety-ai
```

### Option 2: Cloud Platform Deployment

#### Streamlit Cloud

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Configure the deployment settings
4. Deploy with automatic scaling

#### Heroku Deployment

Create necessary files:

`Procfile`:
```
web: streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```

`runtime.txt`:
```
python-3.11.0
```

Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

#### AWS EC2 Deployment

1. Launch an EC2 instance (t3.medium or larger recommended)
2. Install Python and dependencies
3. Configure security groups to allow traffic on port 8501
4. Use a process manager like systemd or supervisor for production

Example systemd service file (`/etc/systemd/system/human-safety-ai.service`):

```ini
[Unit]
Description=Human Safety AI Application
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/human_safety_ai
Environment=PATH=/home/ubuntu/human_safety_ai_env/bin
ExecStart=/home/ubuntu/human_safety_ai_env/bin/streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl enable human-safety-ai
sudo systemctl start human-safety-ai
```

## Configuration Options

### Environment Variables

Configure the application using environment variables:

```bash
# Model configuration
export EMOTION_MODEL_NAME="j-hartmann/emotion-english-distilroberta-base"
export CACHE_DIR="/tmp/model_cache"

# Application settings
export MAX_FILE_SIZE="200MB"
export DEBUG_MODE="false"

# Logging configuration
export LOG_LEVEL="INFO"
export LOG_FILE="/var/log/human_safety_ai.log"
```

### Streamlit Configuration

Create `.streamlit/config.toml` for production:

```toml
[server]
port = 8501
headless = true
enableCORS = true
enableXsrfProtection = true
maxUploadSize = 200

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[logger]
level = "info"
```

### Model Configuration

Customize model settings in `src/config.py`:

```python
import os

# Model configurations
EMOTION_MODEL_NAME = os.getenv("EMOTION_MODEL_NAME", "j-hartmann/emotion-english-distilroberta-base")
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.05"))
CACHE_DIR = os.getenv("CACHE_DIR", "./model_cache")

# Application settings
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "human_safety_ai.log")
```

## Performance Optimization

### Model Caching

Implement efficient model caching to reduce startup time:

```python
import streamlit as st
from functools import lru_cache

@st.cache_resource
def load_emotion_model():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")

@lru_cache(maxsize=128)
def analyze_cached_text(text_hash):
    # Cache analysis results for frequently analyzed texts
    return perform_analysis(text_hash)
```

### Memory Management

Monitor and optimize memory usage:

```python
import gc
import psutil

def monitor_memory():
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024 / 1024,  # MB
        'vms': memory_info.vms / 1024 / 1024   # MB
    }

def cleanup_memory():
    gc.collect()
    # Clear model caches if memory usage is high
    if monitor_memory()['rss'] > 2000:  # 2GB threshold
        clear_model_caches()
```

### Database Integration

For production use, consider storing analysis results:

```python
import sqlite3
from datetime import datetime

class AnalysisLogger:
    def __init__(self, db_path="analysis_log.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                text_hash TEXT,
                sentiment_label TEXT,
                emotion_label TEXT,
                dominant_tone TEXT,
                flag_status TEXT
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_analysis(self, text_hash, results):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT INTO analysis_results 
            (timestamp, text_hash, sentiment_label, emotion_label, dominant_tone, flag_status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            text_hash,
            results['sentiment_label'],
            results['emotion_label'],
            results['dominant_tone'],
            results['flag_status']
        ))
        conn.commit()
        conn.close()
```

## Security Considerations

### Input Validation

Implement robust input validation:

```python
import re
from typing import Optional

def validate_text_input(text: str) -> Optional[str]:
    if not text or not isinstance(text, str):
        return "Invalid input: text must be a non-empty string"
    
    if len(text) > 10000:  # 10K character limit
        return "Input too long: maximum 10,000 characters allowed"
    
    # Check for potentially malicious content
    if re.search(r'<script|javascript:|data:', text, re.IGNORECASE):
        return "Invalid input: potentially malicious content detected"
    
    return None

def validate_audio_file(file) -> Optional[str]:
    if not file:
        return "No file provided"
    
    allowed_extensions = ['.wav', '.mp3', '.m4a']
    if not any(file.name.lower().endswith(ext) for ext in allowed_extensions):
        return f"Invalid file type: only {', '.join(allowed_extensions)} files are allowed"
    
    if file.size > 200 * 1024 * 1024:  # 200MB limit
        return "File too large: maximum 200MB allowed"
    
    return None
```

### Data Privacy

Implement privacy protection measures:

```python
import hashlib
import tempfile
import os

def hash_sensitive_data(text: str) -> str:
    """Create a hash of sensitive text for logging without storing the actual content"""
    return hashlib.sha256(text.encode()).hexdigest()

def secure_temp_file_handling(file_content: bytes) -> str:
    """Securely handle temporary files"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
        temp_file.write(file_content)
        temp_path = temp_file.name
    
    try:
        # Process the file
        result = process_audio_file(temp_path)
        return result
    finally:
        # Always clean up temporary files
        if os.path.exists(temp_path):
            os.remove(temp_path)
```

### Access Control

For production deployments, implement access control:

```python
import streamlit as st
from functools import wraps

def require_authentication(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        
        if not st.session_state.authenticated:
            show_login_form()
            return
        
        return func(*args, **kwargs)
    return wrapper

def show_login_form():
    st.title("Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")
```

## Monitoring and Maintenance

### Logging Configuration

Set up comprehensive logging:

```python
import logging
import sys
from datetime import datetime

def setup_logging(log_level="INFO", log_file=None):
    logger = logging.getLogger("human_safety_ai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Usage
logger = setup_logging("INFO", "human_safety_ai.log")
logger.info("Application started")
```

### Health Monitoring

Implement health checks:

```python
import time
import psutil
from datetime import datetime

class HealthMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def get_health_status(self):
        uptime = time.time() - self.start_time
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent(interval=1)
        
        return {
            'status': 'healthy' if memory_usage < 90 and cpu_usage < 90 else 'degraded',
            'uptime_seconds': uptime,
            'memory_usage_percent': memory_usage,
            'cpu_usage_percent': cpu_usage,
            'total_requests': self.request_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1),
            'timestamp': datetime.now().isoformat()
        }
    
    def record_request(self, success=True):
        self.request_count += 1
        if not success:
            self.error_count += 1

# Add health endpoint to Streamlit app
health_monitor = HealthMonitor()

def show_health_status():
    if st.sidebar.button("Show Health Status"):
        status = health_monitor.get_health_status()
        st.sidebar.json(status)
```

### Performance Metrics

Track application performance:

```python
import time
from contextlib import contextmanager

@contextmanager
def measure_time(operation_name):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{operation_name} completed in {duration:.2f} seconds")

# Usage
with measure_time("Text preprocessing"):
    processed_data = preprocess_text(input_text)

with measure_time("Emotion analysis"):
    emotion_result = analyze_emotion(processed_data['cleaned_text'])
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Model Loading Errors

**Symptoms**: Application fails to start with transformer model errors

**Solutions**:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface/transformers/

# Reinstall transformers
pip uninstall transformers
pip install transformers

# Check internet connectivity for model downloads
curl -I https://huggingface.co/
```

#### Issue: Memory Errors

**Symptoms**: Out of memory errors during model inference

**Solutions**:
```python
# Implement model unloading
def unload_models():
    if 'emotion_model' in globals():
        del emotion_model
    gc.collect()

# Use model quantization (if available)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float16  # Use half precision
)
```

#### Issue: Slow Performance

**Symptoms**: Long response times for analysis

**Solutions**:
```python
# Implement text length limits
MAX_TEXT_LENGTH = 5000

def truncate_text(text, max_length=MAX_TEXT_LENGTH):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

# Use batch processing for multiple requests
def batch_analyze(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)
    return results
```

#### Issue: Audio Processing Errors

**Symptoms**: Speech-to-text conversion fails

**Solutions**:
```bash
# Install additional audio dependencies
sudo apt-get install ffmpeg
pip install pydub[mp3]

# Check audio file format
file audio_file.wav
ffprobe audio_file.wav
```

### Debugging Tools

Enable debug mode for detailed logging:

```python
import streamlit as st

if st.sidebar.checkbox("Debug Mode"):
    st.sidebar.write("Debug Information:")
    st.sidebar.write(f"Python version: {sys.version}")
    st.sidebar.write(f"Streamlit version: {st.__version__}")
    st.sidebar.write(f"Memory usage: {psutil.virtual_memory().percent}%")
    st.sidebar.write(f"CPU usage: {psutil.cpu_percent()}%")
```

## Scaling Considerations

### Horizontal Scaling

For high-traffic deployments, consider:

1. **Load Balancing**: Use nginx or AWS ALB to distribute traffic
2. **Container Orchestration**: Deploy with Kubernetes or Docker Swarm
3. **Database Scaling**: Use distributed databases for analysis logging
4. **Caching**: Implement Redis for shared caching across instances

### Vertical Scaling

Optimize single-instance performance:

1. **GPU Acceleration**: Use CUDA-enabled GPUs for model inference
2. **Model Optimization**: Use model quantization and pruning
3. **Async Processing**: Implement asynchronous request handling
4. **Resource Monitoring**: Continuously monitor and optimize resource usage

### Example Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: human-safety-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: human-safety-ai
  template:
    metadata:
      labels:
        app: human-safety-ai
    spec:
      containers:
      - name: human-safety-ai
        image: human-safety-ai:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: LOG_LEVEL
          value: "INFO"
---
apiVersion: v1
kind: Service
metadata:
  name: human-safety-ai-service
spec:
  selector:
    app: human-safety-ai
  ports:
  - port: 80
    targetPort: 8501
  type: LoadBalancer
```

## Conclusion

This deployment guide provides comprehensive instructions for deploying the Human Safety AI project in various environments. Whether you're setting up a local development environment or deploying to production, following these guidelines will help ensure a stable, secure, and performant deployment.

Remember to regularly update dependencies, monitor system performance, and implement appropriate security measures based on your specific use case and requirements. The AI field evolves rapidly, so stay informed about new developments and best practices in model deployment and system optimization.

For additional support or questions about deployment, refer to the project documentation or consult with experienced DevOps professionals familiar with AI/ML system deployments.

