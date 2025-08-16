# 部署和集成指南

本指南详细介绍了如何在不同环境中部署模型导出优化系统，以及如何将其集成到现有的机器学习工作流中。

## 目录

- [部署环境要求](#部署环境要求)
- [安装和配置](#安装和配置)
- [Docker部署](#docker部署)
- [云平台部署](#云平台部署)
- [CI/CD集成](#cicd集成)
- [生产环境部署](#生产环境部署)
- [监控和维护](#监控和维护)
- [性能优化](#性能优化)
- [安全考虑](#安全考虑)

## 部署环境要求

### 最小系统要求

| 组件 | 最小要求 | 推荐配置 |
|------|----------|----------|
| CPU | 4核心 | 8核心+ |
| 内存 | 16GB | 32GB+ |
| GPU | 8GB VRAM | 24GB+ VRAM |
| 存储 | 100GB SSD | 500GB+ NVMe SSD |
| 网络 | 100Mbps | 1Gbps+ |

### 软件依赖

```bash
# Python环境
Python >= 3.8
pip >= 21.0

# 核心依赖
torch >= 1.12.0
transformers >= 4.21.0
onnx >= 1.12.0
onnxruntime >= 1.12.0

# 可选依赖
tensorrt >= 8.0.0  # TensorRT支持
nvidia-ml-py3      # GPU监控
psutil            # 系统监控
```

### 硬件兼容性

#### GPU支持

```bash
# 检查CUDA版本
nvcc --version
nvidia-smi

# 支持的GPU架构
- Pascal (GTX 10xx系列)
- Turing (RTX 20xx系列)
- Ampere (RTX 30xx, A100系列)
- Ada Lovelace (RTX 40xx系列)
- Hopper (H100系列)
```

#### CPU支持

```bash
# 支持的CPU架构
- x86_64 (Intel/AMD)
- ARM64 (Apple Silicon, ARM服务器)

# 推荐特性
- AVX2指令集支持
- 多核心处理器
- 大容量缓存
```

## 安装和配置

### 标准安装

```bash
# 1. 创建虚拟环境
python -m venv model_export_env
source model_export_env/bin/activate  # Linux/Mac
# 或
model_export_env\Scripts\activate     # Windows

# 2. 升级pip
pip install --upgrade pip

# 3. 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers onnx onnxruntime-gpu

# 4. 安装项目
pip install -e .

# 5. 验证安装
python -c "from src.model_export_controller import ModelExportController; print('安装成功')"
```

### 开发环境安装

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 安装pre-commit钩子
pre-commit install

# 运行测试
pytest tests/
```

### 配置文件设置

```bash
# 创建配置目录
mkdir -p ~/.model_export/

# 创建全局配置文件
cat > ~/.model_export/config.yaml << EOF
default:
  base_model_cache_dir: "~/.cache/huggingface/transformers"
  output_base_dir: "~/exported_models"
  log_level: "INFO"
  max_workers: 2

gpu:
  memory_fraction: 0.8
  allow_growth: true

monitoring:
  enable_metrics: true
  metrics_port: 8080
EOF
```

## Docker部署

### 基础Docker镜像

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY requirements.txt .
COPY src/ ./src/
COPY examples/ ./examples/
COPY docs/ ./docs/

# 安装Python依赖
RUN pip3 install --no-cache-dir -r requirements.txt

# 创建非root用户
RUN useradd -m -u 1000 modelexport
RUN chown -R modelexport:modelexport /app
USER modelexport

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["python3", "-m", "src.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  model-export:
    build: .
    container_name: model-export-service
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./exported_models:/app/exported_models
      - ./logs:/app/logs
      - ~/.cache/huggingface:/home/modelexport/.cache/huggingface
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - EXPORT_LOG_LEVEL=INFO
      - EXPORT_OUTPUT_DIR=/app/exported_models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: model-export-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    container_name: model-export-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - model-export
    restart: unless-stopped

volumes:
  redis_data:
```

### 构建和运行

```bash
# 构建镜像
docker build -t model-export:latest .

# 运行容器
docker-compose up -d

# 查看日志
docker-compose logs -f model-export

# 健康检查
curl http://localhost:8080/health
```

## 云平台部署

### AWS部署

#### EC2实例配置

```bash
# 选择合适的实例类型
# p3.2xlarge  - 1x V100, 8 vCPU, 61GB RAM
# p3.8xlarge  - 4x V100, 32 vCPU, 244GB RAM
# g4dn.xlarge - 1x T4, 4 vCPU, 16GB RAM

# 启动实例
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type p3.2xlarge \
    --key-name your-key-pair \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --user-data file://user-data.sh
```

#### ECS部署

```json
{
  "family": "model-export-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "model-export",
      "image": "your-account.dkr.ecr.region.amazonaws.com/model-export:latest",
      "memory": 15360,
      "cpu": 4096,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "EXPORT_OUTPUT_DIR",
          "value": "/app/exported_models"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "efs-storage",
          "containerPath": "/app/exported_models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/model-export",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "efs-storage",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-xxxxxxxxx"
      }
    }
  ]
}
```

### Google Cloud Platform部署

#### GKE部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-export-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: model-export
  template:
    metadata:
      labels:
        app: model-export
    spec:
      containers:
      - name: model-export
        image: gcr.io/your-project/model-export:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: EXPORT_OUTPUT_DIR
          value: "/app/exported_models"
        volumeMounts:
        - name: model-storage
          mountPath: /app/exported_models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-export-pvc
      nodeSelector:
        accelerator: nvidia-tesla-t4
---
apiVersion: v1
kind: Service
metadata:
  name: model-export-service
spec:
  selector:
    app: model-export
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

### Azure部署

#### AKS部署

```yaml
# azure-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-export
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-export
  template:
    metadata:
      labels:
        app: model-export
    spec:
      containers:
      - name: model-export
        image: your-registry.azurecr.io/model-export:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
            cpu: 8
        ports:
        - containerPort: 8080
        env:
        - name: AZURE_STORAGE_CONNECTION_STRING
          valueFrom:
            secretKeyRef:
              name: azure-storage-secret
              key: connection-string
      nodeSelector:
        accelerator: nvidia
```

## CI/CD集成

### GitHub Actions

```yaml
# .github/workflows/model-export.yml
name: Model Export Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点运行

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  export-model:
    needs: test
    runs-on: self-hosted  # 需要GPU的自托管runner
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Download checkpoint
      run: |
        # 从artifact存储下载最新checkpoint
        aws s3 sync s3://your-bucket/checkpoints/ ./checkpoints/
    
    - name: Export model
      run: |
        python -m src.cli export \
          --config configs/production.yaml \
          --checkpoint ./checkpoints/latest \
          --output ./exported_models
    
    - name: Upload exported models
      run: |
        aws s3 sync ./exported_models s3://your-bucket/exported-models/
    
    - name: Deploy to staging
      run: |
        # 部署到staging环境
        kubectl apply -f k8s/staging/
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - export
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pytest tests/
  coverage: '/TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

export-model:
  stage: export
  image: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  tags:
    - gpu
  script:
    - python -m src.cli export --config configs/production.yaml
  artifacts:
    paths:
      - exported_models/
    expire_in: 1 week
  only:
    - main

deploy-staging:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl apply -f k8s/staging/
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - main

deploy-production:
  stage: deploy
  image: bitnami/kubectl:latest
  script:
    - kubectl apply -f k8s/production/
  environment:
    name: production
    url: https://api.example.com
  when: manual
  only:
    - main
```

### Jenkins Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'model-export'
        KUBECONFIG = credentials('kubeconfig')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            steps {
                sh '''
                    python -m venv venv
                    source venv/bin/activate
                    pip install -r requirements.txt
                    pytest tests/
                '''
            }
        }
        
        stage('Build') {
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push('latest')
                    }
                }
            }
        }
        
        stage('Export Model') {
            agent {
                label 'gpu-node'
            }
            steps {
                sh '''
                    docker run --rm --gpus all \
                        -v $(pwd)/checkpoints:/app/checkpoints \
                        -v $(pwd)/exported_models:/app/exported_models \
                        ${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER} \
                        python -m src.cli export --config configs/production.yaml
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'exported_models/**/*', fingerprint: true
                }
            }
        }
        
        stage('Deploy') {
            steps {
                sh '''
                    kubectl set image deployment/model-export \
                        model-export=${DOCKER_REGISTRY}/${IMAGE_NAME}:${BUILD_NUMBER}
                    kubectl rollout status deployment/model-export
                '''
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Pipeline Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Pipeline failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

## 生产环境部署

### 高可用配置

```yaml
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-export
  namespace: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: model-export
  template:
    metadata:
      labels:
        app: model-export
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - model-export
              topologyKey: kubernetes.io/hostname
      containers:
      - name: model-export
        image: your-registry/model-export:v1.0.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        env:
        - name: EXPORT_LOG_LEVEL
          value: "INFO"
        - name: EXPORT_MAX_WORKERS
          value: "2"
        volumeMounts:
        - name: model-cache
          mountPath: /app/.cache
        - name: exported-models
          mountPath: /app/exported_models
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: exported-models
        persistentVolumeClaim:
          claimName: exported-models-pvc
```

### 负载均衡配置

```yaml
# k8s/production/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-export-service
  namespace: production
spec:
  selector:
    app: model-export
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: model-export-ingress
  namespace: production
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: tls-secret
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: model-export-service
            port:
              number: 80
```

### 存储配置

```yaml
# k8s/production/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: exported-models-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 500Gi
  storageClassName: standard
```

## 监控和维护

### Prometheus监控

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
    - job_name: 'model-export'
      static_configs:
      - targets: ['model-export-service:8080']
      metrics_path: /metrics
      scrape_interval: 30s
    
    - job_name: 'gpu-metrics'
      static_configs:
      - targets: ['gpu-exporter:9400']
      scrape_interval: 10s

    rule_files:
    - "alert_rules.yml"

  alert_rules.yml: |
    groups:
    - name: model-export-alerts
      rules:
      - alert: HighMemoryUsage
        expr: model_export_memory_usage_percent > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is {{ $value }}%"
      
      - alert: GPUMemoryHigh
        expr: nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes > 0.95
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "GPU memory usage critical"
          description: "GPU memory usage is {{ $value | humanizePercentage }}"
      
      - alert: ExportJobFailed
        expr: increase(model_export_jobs_failed_total[5m]) > 0
        labels:
          severity: warning
        annotations:
          summary: "Model export job failed"
          description: "{{ $value }} export jobs failed in the last 5 minutes"
```

### Grafana仪表板

```json
{
  "dashboard": {
    "title": "Model Export System",
    "panels": [
      {
        "title": "Export Jobs",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(model_export_jobs_total[5m])",
            "legendFormat": "Jobs/sec"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization_gpu",
            "legendFormat": "GPU {{ $labels.gpu }}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "model_export_memory_usage_bytes",
            "legendFormat": "Memory Usage"
          }
        ]
      },
      {
        "title": "Export Duration",
        "type": "heatmap",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(model_export_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### 日志聚合

```yaml
# logging/fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/model-export-*.log
      pos_file /var/log/fluentd-model-export.log.pos
      tag kubernetes.model-export
      format json
      time_key time
      time_format %Y-%m-%dT%H:%M:%S.%NZ
    </source>
    
    <filter kubernetes.model-export>
      @type parser
      key_name log
      reserve_data true
      <parse>
        @type json
      </parse>
    </filter>
    
    <match kubernetes.model-export>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name model-export-logs
      type_name _doc
      include_tag_key true
      tag_key @log_name
      flush_interval 1s
    </match>
```

### 健康检查

```python
# src/health_check.py
from flask import Flask, jsonify
import psutil
import torch
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/health')
def health_check():
    """基本健康检查"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })

@app.route('/ready')
def readiness_check():
    """就绪检查"""
    checks = {
        'gpu_available': torch.cuda.is_available(),
        'memory_ok': psutil.virtual_memory().percent < 90,
        'disk_ok': psutil.disk_usage('/').percent < 90
    }
    
    all_ready = all(checks.values())
    
    return jsonify({
        'ready': all_ready,
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat()
    }), 200 if all_ready else 503

@app.route('/metrics')
def metrics():
    """Prometheus指标"""
    gpu_count = torch.cuda.device_count()
    gpu_memory_used = 0
    gpu_memory_total = 0
    
    if gpu_count > 0:
        for i in range(gpu_count):
            gpu_memory_used += torch.cuda.memory_allocated(i)
            gpu_memory_total += torch.cuda.get_device_properties(i).total_memory
    
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    metrics_text = f"""
# HELP model_export_gpu_memory_used_bytes GPU memory used in bytes
# TYPE model_export_gpu_memory_used_bytes gauge
model_export_gpu_memory_used_bytes {gpu_memory_used}

# HELP model_export_gpu_memory_total_bytes GPU memory total in bytes
# TYPE model_export_gpu_memory_total_bytes gauge
model_export_gpu_memory_total_bytes {gpu_memory_total}

# HELP model_export_memory_used_bytes System memory used in bytes
# TYPE model_export_memory_used_bytes gauge
model_export_memory_used_bytes {memory.used}

# HELP model_export_memory_total_bytes System memory total in bytes
# TYPE model_export_memory_total_bytes gauge
model_export_memory_total_bytes {memory.total}

# HELP model_export_disk_used_bytes Disk space used in bytes
# TYPE model_export_disk_used_bytes gauge
model_export_disk_used_bytes {disk.used}

# HELP model_export_disk_total_bytes Disk space total in bytes
# TYPE model_export_disk_total_bytes gauge
model_export_disk_total_bytes {disk.total}
"""
    
    return metrics_text, 200, {'Content-Type': 'text/plain'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## 性能优化

### 资源调优

```yaml
# 生产环境资源配置
resources:
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
    cpu: "4"
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
    cpu: "8"

# 节点选择器
nodeSelector:
  node-type: gpu-optimized
  instance-type: p3.2xlarge

# 容忍度配置
tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

### 缓存策略

```python
# 模型缓存配置
CACHE_CONFIG = {
    'model_cache_dir': '/app/.cache/models',
    'max_cache_size': '50GB',
    'cache_ttl': 86400,  # 24小时
    'cleanup_interval': 3600  # 1小时
}

# Redis缓存配置
REDIS_CONFIG = {
    'host': 'redis.cache.svc.cluster.local',
    'port': 6379,
    'db': 0,
    'max_connections': 20
}
```

### 并发优化

```python
# 并发配置
CONCURRENCY_CONFIG = {
    'max_workers': 2,  # 限制并发导出任务
    'queue_size': 10,  # 任务队列大小
    'timeout': 3600,   # 任务超时时间
    'retry_attempts': 3  # 重试次数
}
```

## 安全考虑

### 网络安全

```yaml
# 网络策略
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: model-export-netpol
spec:
  podSelector:
    matchLabels:
      app: model-export
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP
      port: 80   # HTTP
```

### 访问控制

```yaml
# RBAC配置
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: model-export-role
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: model-export-rolebinding
subjects:
- kind: ServiceAccount
  name: model-export-sa
roleRef:
  kind: Role
  name: model-export-role
  apiGroup: rbac.authorization.k8s.io
```

### 密钥管理

```yaml
# 密钥配置
apiVersion: v1
kind: Secret
metadata:
  name: model-export-secrets
type: Opaque
data:
  huggingface-token: <base64-encoded-token>
  aws-access-key: <base64-encoded-key>
  aws-secret-key: <base64-encoded-secret>
```

### 镜像安全

```dockerfile
# 使用非root用户
RUN useradd -m -u 1000 modelexport
USER modelexport

# 最小化镜像
FROM python:3.9-slim as builder
# ... 构建步骤

FROM python:3.9-slim
COPY --from=builder /app /app
# 只复制必要文件

# 安全扫描
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*
```

## 维护和更新

### 滚动更新

```bash
# 更新部署
kubectl set image deployment/model-export \
    model-export=your-registry/model-export:v1.1.0

# 监控更新状态
kubectl rollout status deployment/model-export

# 回滚更新
kubectl rollout undo deployment/model-export
```

### 备份策略

```bash
# 备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/model-export-$DATE"

# 备份配置
kubectl get configmap model-export-config -o yaml > $BACKUP_DIR/config.yaml

# 备份密钥
kubectl get secret model-export-secrets -o yaml > $BACKUP_DIR/secrets.yaml

# 备份持久卷
kubectl exec -it model-export-pod -- tar czf - /app/exported_models | \
    gzip > $BACKUP_DIR/exported_models.tar.gz
```

### 监控告警

```yaml
# 告警规则
groups:
- name: model-export
  rules:
  - alert: ModelExportDown
    expr: up{job="model-export"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Model export service is down"
  
  - alert: HighErrorRate
    expr: rate(model_export_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate in model export"
```

通过遵循本部署指南，你可以在各种环境中成功部署和运行模型导出优化系统，确保系统的高可用性、可扩展性和安全性。