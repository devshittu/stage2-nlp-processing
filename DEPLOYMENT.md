# Deployment Guide - Stage 2 NLP Processing Service

Comprehensive deployment instructions for production environments.

---

## 📋 **Pre-Deployment Checklist**

### **Hardware Requirements**
- [ ] NVIDIA GPU with 16GB+ VRAM (RTX A4000, A5000, or better)
- [ ] 48+ CPU cores (or adjust `dask_local_cluster_n_workers` in config)
- [ ] 160GB+ RAM (or adjust `dask_cluster_total_memory` in config)
- [ ] 500GB+ SSD storage for models, data, and logs
- [ ] Linux OS (Ubuntu 22.04 LTS recommended)

### **Software Requirements**
- [ ] Docker Engine 20.10+ ([Install Guide](https://docs.docker.com/engine/install/))
- [ ] Docker Compose v2 ([Included in modern Docker](https://docs.docker.com/compose/install/))
- [ ] NVIDIA Driver 525+ ([Install Guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html))
- [ ] NVIDIA Container Toolkit ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### **Account Requirements**
- [ ] HuggingFace account with access token ([Get Token](https://huggingface.co/settings/tokens))
- [ ] Access to required models (all are open-source and freely available)

---

## 🚀 **Step-by-Step Deployment**

### **Step 1: System Preparation**

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify Docker installation
docker --version
docker compose version

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access from Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi
```

### **Step 2: Clone Repository**

```bash
# Clone to target directory
cd /opt  # or your preferred location
git clone <your-repo-url> stage2-nlp-processing
cd stage2-nlp-processing

# Set permissions
sudo chown -R $USER:$USER .
chmod +x run.sh
```

### **Step 3: Configure Environment**

```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Required Settings in `.env`:**
```bash
# HuggingFace token (REQUIRED)
HUGGINGFACE_TOKEN=hf_YOUR_ACTUAL_TOKEN_HERE

# PostgreSQL (if using)
POSTGRES_PASSWORD=generate_secure_password_here

# Elasticsearch (if using)
ELASTICSEARCH_API_KEY=your_api_key_if_needed

# Optional: Adjust service URLs if not using Docker network
NER_SERVICE_URL=http://ner-service:8001
DP_SERVICE_URL=http://dp-service:8002
EVENT_LLM_SERVICE_URL=http://event-llm-service:8003
```

### **Step 4: Configure Application**

Edit `config/settings.yaml` for your environment:

```yaml
# Adjust based on your hardware
celery:
  dask_local_cluster_n_workers: 22  # Set to ~half your CPU cores
  dask_cluster_total_memory: "140GB"  # Leave 20GB for OS

# Enable storage backends you want to use
storage:
  enabled_backends:
    - "jsonl"
    # - "postgresql"  # Uncomment if using PostgreSQL
    # - "elasticsearch"  # Uncomment if using Elasticsearch
```

### **Step 5: Enable PostgreSQL (Optional)**

If using PostgreSQL storage:

1. Uncomment PostgreSQL service in `docker-compose.yml`
2. Set `POSTGRES_PASSWORD` in `.env`
3. Enable in `config/settings.yaml`:
   ```yaml
   storage:
     enabled_backends:
       - "jsonl"
       - "postgresql"
   ```

### **Step 6: Enable Elasticsearch (Optional)**

If using Elasticsearch storage:

1. Uncomment Elasticsearch service in `docker-compose.yml`
2. Enable in `config/settings.yaml`:
   ```yaml
   storage:
     enabled_backends:
       - "jsonl"
       - "elasticsearch"
   ```

### **Step 7: Build Docker Images**

```bash
# Build all services (takes ~15-20 minutes first time)
./run.sh build

# Monitor build progress
docker compose -p nlp-stage2 build --progress=plain
```

**Expected Build Times (first time):**
- NER Service: ~5 minutes
- DP Service: ~5 minutes
- Event LLM Service: ~8 minutes (vLLM compilation)
- Orchestrator: ~10 minutes (includes all models)

### **Step 8: Start Services**

```bash
# Start all services
./run.sh start

# Verify services are starting
docker compose -p nlp-stage2 ps

# Watch logs during startup
./run.sh logs -f
```

**Expected Startup Times:**
- Redis: ~5 seconds
- NER Service: ~30 seconds (model loading)
- DP Service: ~30 seconds (spaCy model loading)
- Event LLM Service: ~60-90 seconds (vLLM + model loading)
- Orchestrator: ~30 seconds
- Celery Worker: ~60 seconds

### **Step 9: Health Check**

```bash
# Check service status
./run.sh status

# Detailed health check via CLI
python -m src.cli.main admin health

# Or via API
curl http://localhost:8000/health | jq '.'
```

**Expected Output:**
```json
{
  "status": "ok",
  "services": {
    "ner_service": {"status": "healthy"},
    "dp_service": {"status": "healthy"},
    "event_llm_service": {"status": "healthy"}
  }
}
```

### **Step 10: Test Processing**

```bash
# Test single document
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "document": {
      "document_id": "test_001",
      "cleaned_text": "President Biden announced new climate policy yesterday."
    }
  }' | jq '.result.events[0]'

# Or using CLI
python -m src.cli.main documents process "President Biden announced new climate policy yesterday."
```

### **Step 11: Test Batch Processing**

```bash
# Submit sample batch
python -m src.cli.main documents batch data/sample_stage1_documents.jsonl

# Monitor job (replace JOB_ID)
python -m src.cli.main jobs status <JOB_ID>
```

---

## 🔧 **Production Optimizations**

### **1. Resource Limits**

Edit `docker-compose.yml` to set resource limits:

```yaml
orchestrator:
  deploy:
    resources:
      limits:
        cpus: '24'
        memory: 140G
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### **2. Persistent Storage**

Ensure volumes are properly configured:

```yaml
volumes:
  redis_data:
    driver: local
  huggingface_cache:
    driver: local
  # For production, consider using named volumes on separate disks
  data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/data/nlp-stage2
```

### **3. Logging Configuration**

For production logging:

```yaml
# config/settings.yaml
monitoring:
  log_level: "INFO"  # Use INFO, not DEBUG in production
  log_file: "/app/logs/nlp_processing.log"
  log_rotation: "100MB"
  log_retention_days: 30
```

Set up log rotation:

```bash
# Create logrotate config
sudo nano /etc/logrotate.d/nlp-stage2

# Add:
/opt/stage2-nlp-processing/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
}
```

### **4. Monitoring & Alerts**

Enable Prometheus metrics:

```yaml
# config/settings.yaml
monitoring:
  enable_metrics: true
  metrics_port: 9090
```

Access metrics at: `http://localhost:9090/metrics`

### **5. Backup Strategy**

```bash
# Backup script (run daily via cron)
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR="/mnt/backups/nlp-stage2/$DATE"

mkdir -p $BACKUP_DIR

# Backup data
tar -czf $BACKUP_DIR/data.tar.gz /opt/stage2-nlp-processing/data

# Backup config
cp -r /opt/stage2-nlp-processing/config $BACKUP_DIR/

# Backup Docker volumes
docker run --rm -v nlp-stage2_redis_data:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar -czf /backup/redis_data.tar.gz -C /data .

echo "Backup completed: $BACKUP_DIR"
```

---

## 🌐 **Reverse Proxy Setup (Production)**

### **Using Nginx**

```nginx
# /etc/nginx/sites-available/nlp-stage2

upstream orchestrator {
    server localhost:8000;
}

server {
    listen 80;
    server_name nlp.yourdomain.com;

    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name nlp.yourdomain.com;

    # SSL configuration
    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy settings
    location / {
        proxy_pass http://orchestrator;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long-running requests
        proxy_read_timeout 600s;
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
    }

    # Metrics endpoint (restrict access)
    location /metrics {
        allow 10.0.0.0/8;  # Adjust to your monitoring network
        deny all;
        proxy_pass http://orchestrator;
    }
}
```

Enable and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/nlp-stage2 /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔒 **Security Hardening**

### **1. Firewall Configuration**

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### **2. Environment Variables Protection**

```bash
# Restrict .env file permissions
chmod 600 .env
```

### **3. Container Security**

Add to `docker-compose.yml`:

```yaml
services:
  orchestrator:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
```

### **4. Regular Updates**

```bash
# Update base images monthly
./run.sh rebuild-no-cache

# Update Python packages
pip install -r requirements.txt --upgrade
```

---

## 📊 **Scaling Strategies**

### **Horizontal Scaling (Multiple Nodes)**

For processing >1000 documents/hour:

1. **Deploy orchestrator + workers on separate nodes**
2. **Use external Redis cluster**
3. **Load balance orchestrator with Nginx/HAProxy**
4. **Shared storage for JSONL backend (NFS/S3)**

### **Vertical Scaling (Single Node)**

Adjust worker count based on hardware:

```yaml
# config/settings.yaml
celery:
  dask_local_cluster_n_workers: 32  # Increase if you have more cores
  dask_local_cluster_memory_limit: "7GB"  # Adjust per worker
```

---

## 🔍 **Monitoring & Maintenance**

### **Daily Tasks**

```bash
# Check service health
./run.sh status

# Review logs for errors
./run.sh logs | grep ERROR

# Check disk space
df -h
du -sh /opt/stage2-nlp-processing/data
```

### **Weekly Tasks**

```bash
# Rotate logs
sudo logrotate /etc/logrotate.d/nlp-stage2

# Backup data
./backup.sh

# Check Docker resource usage
docker stats
```

### **Monthly Tasks**

```bash
# Update Docker images
./run.sh rebuild-no-cache

# Review and archive old data
find /opt/stage2-nlp-processing/data -name "*.jsonl" -mtime +90 -exec gzip {} \;

# Check GPU health
nvidia-smi -q | grep -A 10 "Temperature"
```

---

## 🆘 **Disaster Recovery**

### **Backup Everything**

```bash
# 1. Stop services
./run.sh stop

# 2. Backup volumes
docker run --rm -v nlp-stage2_redis_data:/data \
  -v /mnt/backups:/backup alpine \
  tar -czf /backup/redis_$(date +%Y%m%d).tar.gz -C /data .

# 3. Backup configuration
tar -czf /mnt/backups/config_$(date +%Y%m%d).tar.gz config/

# 4. Backup data
tar -czf /mnt/backups/data_$(date +%Y%m%d).tar.gz data/

# 5. Restart services
./run.sh start
```

### **Restore from Backup**

```bash
# 1. Stop services
./run.sh down

# 2. Restore configuration
tar -xzf /mnt/backups/config_YYYYMMDD.tar.gz

# 3. Restore data
tar -xzf /mnt/backups/data_YYYYMMDD.tar.gz

# 4. Restore Redis volume
docker volume create nlp-stage2_redis_data
docker run --rm -v nlp-stage2_redis_data:/data \
  -v /mnt/backups:/backup alpine \
  tar -xzf /backup/redis_YYYYMMDD.tar.gz -C /data

# 5. Rebuild and start
./run.sh rebuild
./run.sh start
```

---

## 📞 **Support & Troubleshooting**

### **Common Production Issues**

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| High memory usage | `docker stats` | Reduce `dask_local_cluster_n_workers` |
| GPU OOM | `nvidia-smi` | Reduce `gpu_memory_utilization` to 0.80 |
| Slow processing | Check logs | Increase `dask_local_cluster_n_workers` |
| Redis connection errors | `docker logs nlp-redis` | Increase Redis memory limit |
| Disk full | `df -h` | Archive/compress old JSONL files |

### **Log Locations**

- Application logs: `/opt/stage2-nlp-processing/logs/`
- Docker logs: `docker logs <container-name>`
- System logs: `/var/log/syslog`

### **Emergency Contacts**

- System Administrator: [admin@yourdomain.com]
- DevOps Team: [devops@yourdomain.com]
- On-Call: [oncall@yourdomain.com]

---

**Deployment Guide Version 1.0 | Last Updated: 2024-11-19**
