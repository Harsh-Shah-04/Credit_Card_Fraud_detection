# 🚀 Deployment Guide - Credit Card Fraud Detection Streamlit App

This guide provides step-by-step instructions to deploy your fraud detection application on various platforms.

## Table of Contents
1. [Local Deployment](#local-deployment)
2. [Streamlit Cloud (Recommended)](#streamlit-cloud-recommended)
3. [Docker Deployment](#docker-deployment)
4. [Heroku Deployment](#heroku-deployment)
5. [AWS Deployment](#aws-deployment)
6. [Azure Deployment](#azure-deployment)

---

## Local Deployment

### Windows
1. Open Command Prompt or PowerShell
2. Navigate to project directory
3. Double-click `run_app.bat` OR run:
   ```bash
   .\run_app.bat
   ```

### MacOS/Linux
1. Open Terminal
2. Navigate to project directory
3. Make script executable:
   ```bash
   chmod +x run_app.sh
   ```
4. Run:
   ```bash
   ./run_app.sh
   ```

### Manual Setup (All Platforms)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# MacOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run streamlit_app.py
```

**Access:** http://localhost:8501

---

## Streamlit Cloud (Recommended)

### Easiest & Fastest Deployment (Free Tier Available)

#### Step 1: Prepare Repository
```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit changes
git commit -m "Initial commit - Fraud Detection App"
```

#### Step 2: Push to GitHub
1. Create a new repository on [GitHub](https://github.com/new)
2. Add remote:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/fraud-detection.git
   git branch -M main
   git push -u origin main
   ```

#### Step 3: Deploy on Streamlit Cloud
1. Visit https://share.streamlit.io
2. Click "New app"
3. Select your GitHub repository
4. Choose:
   - Repository: `YOUR_USERNAME/fraud-detection`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
5. Click "Deploy"

#### Step 4: Monitor Deployment
- Streamlit will build and deploy your app
- Once complete, you'll get a public URL like: `https://your-app-name.streamlit.app`
- Share this URL with others!

#### Advantages:
✅ Free (up to 3 apps)  
✅ Automatic deployments on git push  
✅ Built-in SSL/HTTPS  
✅ No server management  
✅ Scalable  

---

## Docker Deployment

### Prerequisites
- Docker installed on your system
- Docker Hub account (for cloud registry)

### Local Docker Deployment

#### Step 1: Build Image
```bash
docker build -t fraud-detection:latest .
```

#### Step 2: Run Container
```bash
docker run -p 8501:8501 fraud-detection:latest
```

**Access:** http://localhost:8501

### Docker Compose (Easier)

#### Step 1: Run with Compose
```bash
docker-compose up --build
```

#### Step 2: Stop Service
```bash
docker-compose down
```

### Push to Docker Hub (For Cloud Deployment)

```bash
# Login to Docker Hub
docker login

# Tag image
docker tag fraud-detection:latest YOUR_USERNAME/fraud-detection:latest

# Push to Docker Hub
docker push YOUR_USERNAME/fraud-detection:latest

# Deploy on cloud platforms using this image
# Example with any cloud provider's container service
```

---

## Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Heroku account created

### Step 1: Create Heroku App
```bash
heroku login
heroku create your-fraud-detection-app
```

### Step 2: Create Procfile (Already in repo)
File: `Procfile`
```
web: streamlit run --server.port $PORT --server.address 0.0.0.0 streamlit_app.py
```

### Step 3: Deploy
```bash
git push heroku main
```

### Step 4: View Logs
```bash
heroku logs --tail
```

**Access:** https://your-fraud-detection-app.herokuapp.com

#### Advantages:
✅ Easy deployment  
✅ Free tier available  
✅ Good for small projects  
❌ Slow cold starts  

---

## AWS Deployment

### Option 1: AWS Elastic Beanstalk (Easiest)

#### Prerequisites
- AWS account
- AWS CLI installed

#### Step 1: Initialize EB
```bash
pip install awsebcli --upgrade --user

eb init -p python-3.11 fraud-detection
eb create fraud-detection-env
```

#### Step 2: Configure nginx
Create `.ebextensions/streamlit.config`:
```
option_settings:
  aws:elasticbeanstalk:application:environment:
    STREAMLIT_SERVER_HEADLESS: "true"
    STREAMLIT_SERVER_PORT: "8501"
    STREAMLIT_SERVER_ADDRESS: "0.0.0.0"
```

#### Step 3: Deploy
```bash
eb deploy
```

### Option 2: AWS Lambda + API Gateway (Serverless)
Use serverless framework or AWS SAM

### Option 3: EC2 (Full Control)

#### Step 1: Launch EC2 Instance
1. Choose Ubuntu 22.04 LTS
2. Create security group (allow traffic on ports 80, 443, 8501)
3. Create and download key pair

#### Step 2: Connect to Instance
```bash
ssh -i "your-key.pem" ubuntu@your-instance-ip
```

#### Step 3: Install Dependencies
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv -y

# Clone your repository
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection

# Setup and run
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

## Azure Deployment

### Azure Container Instances (Easiest)

#### Step 1: Push to Azure Container Registry
```bash
# Create container registry
az acr create --resource-group myResourceGroup \
  --name frauddetectionregistry --sku Basic

# Login to registry
az acr login --name frauddetectionregistry

# Build and push
az acr build --registry frauddetectionregistry \
  --image fraud-detection:latest .
```

#### Step 2: Deploy Container
```bash
az container create \
  --resource-group myResourceGroup \
  --name fraud-detection-app \
  --image frauddetectionregistry.azurecr.io/fraud-detection:latest \
  --ports 8501 \
  --cpu 1 --memory 1 \
  --registry-login-server frauddetectionregistry.azurecr.io \
  --registry-username yourUsername \
  --registry-password yourPassword
```

### Azure App Service

#### Step 1: Create App Service Plan
```bash
az appservice plan create \
  --name fraudDetectionPlan \
  --resource-group myResourceGroup \
  --sku B1 --is-linux
```

#### Step 2: Create Web App
```bash
az webapp create \
  --resource-group myResourceGroup \
  --plan fraudDetectionPlan \
  --name fraud-detection-app \
  --runtime "PYTHON:3.11"
```

#### Step 3: Deploy
```bash
az webapp up \
  --resource-group myResourceGroup \
  --name fraud-detection-app \
  --runtime "PYTHON:3.11"
```

---

## GCP Deployment

### Cloud Run (Recommended)

#### Step 1: Push to Google Container Registry
```bash
# Set project ID
export PROJECT_ID=$(gcloud config get-value project)

# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/fraud-detection

```

#### Step 2: Deploy to Cloud Run
```bash
gcloud run deploy fraud-detection \
  --image gcr.io/$PROJECT_ID/fraud-detection \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

---

## Performance Optimization Tips

### 1. Model Caching
- Already implemented with `@st.cache_resource`
- Models loaded once per session

### 2. Streamlit Configuration
In `.streamlit/config.toml`:
```toml
[client]
showErrorDetails = false
toolbarMode = "viewer"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = true
```

### 3. Environment Variables
```bash
# For deployment, set:
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_LOGGER_LEVEL=warning
STREAMLIT_CLIENT_TOOLBAR_MODE=viewer
```

---

## Monitoring & Logging

### Streamlit Cloud
- Built-in monitoring dashboard
- View logs in Streamlit Cloud console

### Docker
```bash
# View logs
docker logs -f container_id

# Monitor resource usage
docker stats
```

### Heroku
```bash
# View logs
heroku logs --tail --app your-app-name
```

### AWS
- Use CloudWatch for logs
- Set up alarms for errors

---

## SSL/HTTPS Setup

### If Using Your Own Domain

#### Streamlit Cloud
- HTTPS enabled by default

#### Docker/Self-Hosted
- Use Nginx with Let's Encrypt
- Use AWS ALB/CloudFront with SSL certificates

### Example Nginx Configuration
```nginx
server {
    listen 443 ssl;
    server_name fraud-detection.example.com;

    ssl_certificate /etc/letsencrypt/live/fraud-detection.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/fraud-detection.example.com/privkey.pem;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

---

## Troubleshooting Deployment Issues

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: Port already in use
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

### Issue: Model files not loading
- Ensure models/ directory is in the same location as app.py
- Check file permissions
- For Docker, ensure COPY command includes models/

### Issue: High Memory Usage
- Reduce batch prediction size
- Use smaller models if needed
- Increase container memory allocation

---

## Cost Estimation

| Platform | Free Tier | Estimated Cost |
|----------|-----------|-----------------|
| Streamlit Cloud | Up to 3 apps | $0-5/month |
| Heroku | Deprecated | $7-50/month |
| AWS Elastic Beanstalk | 750 hrs/month EC2 | $5-30/month |
| Docker Hub | Free | $0-12/month |
| Azure Container Instances | 250000 vCPU-s/month | $0-20/month |
| GCP Cloud Run | 2M requests/month free | $0-15/month |

---

## Recommended Setup for Production

**Best Choice: Streamlit Cloud**
- ✅ Easiest to set up
- ✅ Free tier with 3 apps
- ✅ Automatic HTTPS
- ✅ One-click deployments
- ✅ Built-in monitoring

**For Advanced Users: Docker + AWS/GCP**
- ✅ Full control
- ✅ Scalable
- ✅ Better performance
- ⚠️ More complex setup

---

## Next Steps

1. **Test Locally** - Ensure app works on your machine
2. **Choose Platform** - Pick your deployment platform
3. **Deploy** - Follow platform-specific instructions
4. **Monitor** - Set up monitoring and logging
5. **Share** - Share your app URL with stakeholders

---

## Support & Resources

- **Streamlit Docs**: https://docs.streamlit.io/
- **Streamlit Cloud**: https://share.streamlit.io/
- **Docker Docs**: https://docs.docker.com/
- **AWS Docs**: https://docs.aws.amazon.com/
- **Azure Docs**: https://docs.microsoft.com/azure/

---

Happy Deploying! 🚀
