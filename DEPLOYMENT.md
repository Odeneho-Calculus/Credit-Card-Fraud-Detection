# Deployment Guide - Credit Card Fraud Detection System

This guide covers various deployment options for the fraud detection web application.

## 🚀 Quick Local Deployment

### Prerequisites
- Python 3.8+
- All dependencies installed (`pip install -r requirements.txt`)
- Trained models in the `models/` directory

### Start the Application
```bash
# Option 1: Using the startup script (recommended)
python start_api.py

# Option 2: Direct Flask app
python app.py

# Option 3: Using Gunicorn (production-like)
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Access the application at: **http://localhost:5000**

## 🐳 Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t fraud-detection .

# Run the container
docker run -p 5000:5000 fraud-detection

# Or use docker-compose
docker-compose up -d
```

### Docker with Volume Mounting
```bash
# Mount local directories for persistence
docker run -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plots:/app/plots \
  fraud-detection
```

## ☁️ Cloud Deployment

### Heroku
1. Create a `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create your-fraud-detection-app
git push heroku main
```

### AWS EC2
1. Launch an EC2 instance (Ubuntu 20.04 LTS)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip nginx
pip3 install -r requirements.txt
```

3. Configure nginx:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

4. Start the application:
```bash
gunicorn -w 4 -b 127.0.0.1:5000 app:app
```

### Google Cloud Platform
1. Create `app.yaml`:
```yaml
runtime: python39

env_variables:
  FLASK_ENV: production

automatic_scaling:
  min_instances: 1
  max_instances: 10
```

2. Deploy:
```bash
gcloud app deploy
```

## 🔧 Production Configuration

### Environment Variables
```bash
export FLASK_ENV=production
export FLASK_DEBUG=False
export SECRET_KEY=your-secret-key-here
export MODEL_PATH=/path/to/models
```

### Gunicorn Configuration
Create `gunicorn.conf.py`:
```python
bind = "0.0.0.0:5000"
workers = 4
worker_class = "sync"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True
```

### Nginx Configuration
```nginx
upstream fraud_detection {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 10M;

    location / {
        proxy_pass http://fraud_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /path/to/your/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## 📊 Monitoring & Logging

### Health Checks
The application includes a health check endpoint:
```bash
curl http://localhost:5000/api/health
```

### Logging Configuration
Add to your production environment:
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/fraud_detection.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
```

### Performance Monitoring
Consider integrating:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **New Relic** or **DataDog** for APM

## 🔒 Security Considerations

### HTTPS Configuration
Always use HTTPS in production:
```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

### Rate Limiting
Implement rate limiting to prevent abuse:
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["1000 per hour"]
)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("100 per minute")
def predict():
    # Your prediction logic
    pass
```

### API Authentication
For production APIs, consider implementing:
- API keys
- JWT tokens
- OAuth 2.0

## 🧪 Testing Deployment

### Automated Testing
```bash
# Run API tests
python test_api.py

# Load testing with Apache Bench
ab -n 1000 -c 10 http://localhost:5000/api/health

# Load testing with curl
for i in {1..100}; do
  curl -X POST http://localhost:5000/api/predict \
    -H "Content-Type: application/json" \
    -d @sample_transaction.json
done
```

### Performance Benchmarks
Expected performance metrics:
- **Response Time**: < 100ms for single predictions
- **Throughput**: 1000+ requests per second
- **Memory Usage**: < 512MB per worker
- **CPU Usage**: < 50% under normal load

## 🔄 Continuous Deployment

### GitHub Actions
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Production

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Deploy to server
      run: |
        # Your deployment script here
        ssh user@server 'cd /app && git pull && docker-compose up -d'
```

## 📋 Deployment Checklist

- [ ] All dependencies installed
- [ ] Models trained and available
- [ ] Environment variables configured
- [ ] Database connections tested (if applicable)
- [ ] SSL certificates installed
- [ ] Monitoring and logging configured
- [ ] Health checks working
- [ ] Load balancer configured
- [ ] Backup strategy in place
- [ ] Security measures implemented
- [ ] Performance testing completed

## 🆘 Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check if model files exist
   ls -la models/

   # Verify file permissions
   chmod 644 models/*.pkl
   ```

2. **Port already in use**
   ```bash
   # Find process using port 5000
   lsof -i :5000

   # Kill the process
   kill -9 <PID>
   ```

3. **Memory issues**
   ```bash
   # Monitor memory usage
   htop

   # Reduce number of workers
   gunicorn -w 2 app:app
   ```

4. **Slow predictions**
   - Check model loading time
   - Optimize data preprocessing
   - Consider model caching
   - Use faster hardware

### Logs Location
- Application logs: `logs/fraud_detection.log`
- Access logs: `/var/log/nginx/access.log`
- Error logs: `/var/log/nginx/error.log`

## 📞 Support

For deployment issues:
1. Check the logs first
2. Verify all dependencies are installed
3. Test the API endpoints manually
4. Check system resources (CPU, memory, disk)
5. Review security settings

---

**Note**: This deployment guide covers common scenarios. Adjust configurations based on your specific infrastructure and requirements.