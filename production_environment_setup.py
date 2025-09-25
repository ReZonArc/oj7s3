#!/usr/bin/env python3
"""
Production Environment Setup for SKZ Autonomous Agents
======================================================

This script sets up a comprehensive production environment configuration
with all required APIs, databases, and services properly configured.

Usage:
    python3 production_environment_setup.py [--template-only] [--validate]
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any

class ProductionEnvironmentSetup:
    """Setup production environment configuration"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_dir = self.repo_path / "config"
        
    def setup_production_environment(self, template_only: bool = False):
        """Setup comprehensive production environment"""
        print("🚀 Setting up Production Environment for SKZ Autonomous Agents...")
        
        # Create configuration directory
        self.config_dir.mkdir(exist_ok=True)
        
        # Create production configuration templates
        self._create_production_config_template()
        self._create_environment_template()
        self._create_docker_compose_production()
        self._create_nginx_configuration()
        self._create_systemd_services()
        self._create_database_schema()
        
        if not template_only:
            # Set up development environment variables for testing
            self._setup_development_environment()
        
        print("✅ Production environment setup complete!")
        self._print_setup_instructions()
    
    def _create_production_config_template(self):
        """Create comprehensive production configuration template"""
        config = {
            "environment": "production",
            "debug": False,
            "logging": {
                "level": "INFO",
                "format": "structured",
                "output": ["file", "stdout"],
                "file_path": "/var/log/skz-agents/app.log",
                "max_file_size": "100MB",
                "backup_count": 7
            },
            "database": {
                "main": {
                    "url": "${DATABASE_URL}",
                    "pool_size": 20,
                    "max_overflow": 30,
                    "pool_timeout": 30,
                    "pool_recycle": 3600
                },
                "ojs": {
                    "url": "${OJS_DATABASE_URL}",
                    "pool_size": 10,
                    "max_overflow": 15
                },
                "events": {
                    "url": "${EVENTS_DATABASE_URL}",
                    "pool_size": 5,
                    "max_overflow": 10
                }
            },
            "redis": {
                "url": "${REDIS_URL}",
                "timeout": 30,
                "retry_on_timeout": True,
                "max_connections": 50
            },
            "apis": {
                "uspto": {
                    "api_key": "${USPTO_API_KEY}",
                    "base_url": "https://developer.uspto.gov/api",
                    "timeout": 60,
                    "rate_limit": {
                        "requests_per_minute": 60,
                        "burst_limit": 10
                    }
                },
                "google_patents": {
                    "api_key": "${GOOGLE_PATENTS_API_KEY}",
                    "project_id": "${GOOGLE_CLOUD_PROJECT_ID}",
                    "timeout": 45,
                    "rate_limit": {
                        "requests_per_minute": 100,
                        "burst_limit": 15
                    }
                },
                "epo": {
                    "api_key": "${EPO_API_KEY}",
                    "base_url": "https://ops.epo.org/3.2",
                    "timeout": 60
                }
            },
            "communication": {
                "email": {
                    "provider": "sendgrid",
                    "sendgrid": {
                        "api_key": "${SENDGRID_API_KEY}",
                        "from_address": "${EMAIL_FROM_ADDRESS}",
                        "timeout": 30
                    },
                    "smtp": {
                        "enabled": False,
                        "host": "${SMTP_HOST}",
                        "port": "${SMTP_PORT}",
                        "username": "${SMTP_USERNAME}",
                        "password": "${SMTP_PASSWORD}",
                        "use_tls": True
                    }
                },
                "sms": {
                    "provider": "twilio",
                    "twilio": {
                        "account_sid": "${TWILIO_ACCOUNT_SID}",
                        "auth_token": "${TWILIO_AUTH_TOKEN}",
                        "from_number": "${TWILIO_FROM_NUMBER}"
                    }
                }
            },
            "ml": {
                "bert": {
                    "model_name": "bert-base-uncased",
                    "model_path": "${BERT_MODEL_PATH}",
                    "cache_dir": "/var/cache/skz-agents/models",
                    "max_length": 512,
                    "batch_size": 16
                },
                "huggingface": {
                    "api_key": "${HUGGINGFACE_API_KEY}",
                    "cache_dir": "/var/cache/skz-agents/huggingface"
                },
                "inference": {
                    "device": "auto",  # "cpu", "cuda", "auto"
                    "precision": "fp16",
                    "optimize": True
                }
            },
            "security": {
                "jwt": {
                    "secret_key": "${JWT_SECRET_KEY}",
                    "algorithm": "HS256",
                    "expire_minutes": 60
                },
                "cors": {
                    "origins": ["${FRONTEND_URL}"],
                    "credentials": True
                },
                "rate_limiting": {
                    "enabled": True,
                    "default_rate": "1000/hour",
                    "burst_rate": "100/minute"
                }
            },
            "monitoring": {
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "path": "/metrics"
                },
                "health_checks": {
                    "enabled": True,
                    "port": 8080,
                    "path": "/health"
                },
                "sentry": {
                    "dsn": "${SENTRY_DSN}",
                    "environment": "production"
                }
            },
            "features": {
                "use_production_apis": True,
                "enable_caching": True,
                "enable_async_processing": True,
                "max_retries": 3,
                "retry_delay": 30
            }
        }
        
        config_file = self.config_dir / "production.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"📄 Created production configuration template: {config_file}")
    
    def _create_environment_template(self):
        """Create comprehensive environment variables template"""
        env_content = '''# SKZ Autonomous Agents - Production Environment Configuration
# Copy this file to .env and configure all required values

# ============================================================
# ENVIRONMENT CONFIGURATION
# ============================================================
ENVIRONMENT=production
DEBUG=false
SKZ_CONFIG_FILE=config/production.json

# ============================================================
# DATABASE CONFIGURATION
# ============================================================
# Main PostgreSQL database for SKZ agents
DATABASE_URL=postgresql://skz_user:secure_password@localhost:5432/skz_production

# OJS database connection (if integrating with existing OJS)
OJS_DATABASE_URL=postgresql://ojs_user:ojs_password@localhost:5432/ojs_database

# Events/audit database for transaction logging
EVENTS_DATABASE_URL=postgresql://events_user:events_password@localhost:5432/skz_events

# ============================================================
# REDIS CONFIGURATION
# ============================================================
REDIS_URL=redis://localhost:6379/0

# ============================================================
# EXTERNAL API CONFIGURATIONS
# ============================================================
# USPTO Patent API
# Get from: https://developer.uspto.gov/
USPTO_API_KEY=your_uspto_api_key_here

# Google Patents API (requires Google Cloud account)
# Get from: https://console.cloud.google.com/
GOOGLE_PATENTS_API_KEY=your_google_patents_api_key_here
GOOGLE_CLOUD_PROJECT_ID=your_project_id_here

# European Patents Office API
# Get from: https://developers.epo.org/
EPO_API_KEY=your_epo_api_key_here

# ============================================================
# COMMUNICATION SERVICE CONFIGURATIONS
# ============================================================
# SendGrid for email delivery
# Get from: https://sendgrid.com/
SENDGRID_API_KEY=your_sendgrid_api_key_here
EMAIL_FROM_ADDRESS=noreply@yourjournal.com

# Twilio for SMS delivery
# Get from: https://www.twilio.com/
TWILIO_ACCOUNT_SID=your_twilio_account_sid_here
TWILIO_AUTH_TOKEN=your_twilio_auth_token_here
TWILIO_FROM_NUMBER=+1234567890

# SMTP fallback (optional)
SMTP_HOST=smtp.yourprovider.com
SMTP_PORT=587
SMTP_USERNAME=your_smtp_username
SMTP_PASSWORD=your_smtp_password

# ============================================================
# ML MODEL CONFIGURATIONS
# ============================================================
# BERT model path (local path or HuggingFace model name)
BERT_MODEL_PATH=/var/models/bert-base-uncased

# HuggingFace API for model downloads
# Get from: https://huggingface.co/
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# ============================================================
# SECURITY CONFIGURATIONS
# ============================================================
# JWT secret key (generate with: python -c "import secrets; print(secrets.token_hex(32))")
JWT_SECRET_KEY=your_jwt_secret_key_here

# Frontend URL for CORS
FRONTEND_URL=https://yourjournal.com

# ============================================================
# MONITORING AND LOGGING
# ============================================================
# Sentry for error tracking
# Get from: https://sentry.io/
SENTRY_DSN=your_sentry_dsn_here

# Log level
LOG_LEVEL=INFO

# ============================================================
# DEPLOYMENT CONFIGURATION
# ============================================================
# Server configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4

# SSL/TLS (for production)
SSL_CERT_PATH=/etc/ssl/certs/yourjournal.com.crt
SSL_KEY_PATH=/etc/ssl/private/yourjournal.com.key
'''
        
        env_file = self.repo_path / ".env.production.template"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"🔐 Created environment template: {env_file}")
    
    def _create_docker_compose_production(self):
        """Create Docker Compose configuration for production"""
        docker_compose = '''version: '3.8'

services:
  # SKZ Agents Application
  skz-agents:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "5000:5000"
    environment:
      - ENVIRONMENT=production
    env_file:
      - .env.production
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - skz-models:/var/models
    depends_on:
      - postgres-main
      - postgres-events
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Main PostgreSQL Database
  postgres-main:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: skz_production
      POSTGRES_USER: skz_user
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-main-data:/var/lib/postgresql/data
      - ./schema/main.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Events PostgreSQL Database
  postgres-events:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: skz_events
      POSTGRES_USER: events_user
      POSTGRES_PASSWORD: ${EVENTS_POSTGRES_PASSWORD}
    volumes:
      - postgres-events-data:/var/lib/postgresql/data
      - ./schema/events.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
    ports:
      - "5433:5432"
    restart: unless-stopped

  # Redis for caching and distributed locks
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - skz-agents
    restart: unless-stopped

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Grafana dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres-main-data:
  postgres-events-data:
  redis-data:
  prometheus-data:
  grafana-data:
  skz-models:
'''
        
        docker_file = self.repo_path / "docker-compose.production.yml"
        with open(docker_file, 'w') as f:
            f.write(docker_compose)
        
        print(f"🐳 Created Docker Compose production configuration: {docker_file}")
    
    def _create_nginx_configuration(self):
        """Create Nginx configuration for production"""
        nginx_dir = self.repo_path / "nginx"
        nginx_dir.mkdir(exist_ok=True)
        
        nginx_config = '''# SKZ Agents Production Nginx Configuration

upstream skz_agents {
    server skz-agents:5000 max_fails=3 fail_timeout=30s;
}

# Rate limiting
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=5r/s;

server {
    listen 80;
    server_name yourjournal.com www.yourjournal.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourjournal.com www.yourjournal.com;
    
    # SSL Configuration
    ssl_certificate /etc/nginx/ssl/yourjournal.com.crt;
    ssl_certificate_key /etc/nginx/ssl/yourjournal.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Logging
    access_log /var/log/nginx/skz-agents-access.log;
    error_log /var/log/nginx/skz-agents-error.log;
    
    # Rate limiting for API endpoints
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;
        limit_req_status 429;
        
        proxy_pass http://skz_agents;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # Long timeout for ML operations
        
        # Buffer settings
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Authentication endpoints with stricter rate limiting
    location /api/auth/ {
        limit_req zone=auth_limit burst=10 nodelay;
        limit_req_status 429;
        
        proxy_pass http://skz_agents;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Health check endpoint (no rate limiting)
    location /health {
        proxy_pass http://skz_agents;
        proxy_set_header Host $host;
        access_log off;
    }
    
    # Static files
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Frontend application
    location / {
        proxy_pass http://skz_agents;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
'''
        
        nginx_config_file = nginx_dir / "nginx.conf"
        with open(nginx_config_file, 'w') as f:
            f.write(nginx_config)
        
        print(f"🌐 Created Nginx configuration: {nginx_config_file}")
    
    def _create_systemd_services(self):
        """Create systemd service files for production deployment"""
        systemd_dir = self.repo_path / "systemd"
        systemd_dir.mkdir(exist_ok=True)
        
        service_content = '''[Unit]
Description=SKZ Autonomous Agents Framework
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=skz-agents
Group=skz-agents
WorkingDirectory=/opt/skz-agents
Environment=PATH=/opt/skz-agents/venv/bin
ExecStart=/opt/skz-agents/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 5000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=skz-agents

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/skz-agents/logs /var/cache/skz-agents

# Resource limits
LimitNOFILE=65536
MemoryMax=2G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
'''
        
        service_file = systemd_dir / "skz-agents.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print(f"⚙️ Created systemd service: {service_file}")
    
    def _create_database_schema(self):
        """Create database schema files"""
        schema_dir = self.repo_path / "schema"
        schema_dir.mkdir(exist_ok=True)
        
        # Main database schema
        main_schema = '''-- SKZ Agents Main Database Schema

-- Entities table for data synchronization
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    entity_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    data JSONB NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(entity_type, entity_id)
);

-- Create indexes
CREATE INDEX idx_entities_type_id ON entities(entity_type, entity_id);
CREATE INDEX idx_entities_created_at ON entities(created_at);
CREATE INDEX idx_entities_checksum ON entities(checksum);

-- Recipients table for communication automation
CREATE TABLE IF NOT EXISTS recipients (
    recipient_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),
    preferred_communication VARCHAR(50) DEFAULT 'email',
    timezone VARCHAR(50) DEFAULT 'UTC',
    language VARCHAR(10) DEFAULT 'en',
    role VARCHAR(100),
    organization VARCHAR(255),
    communication_preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML models metadata
CREATE TABLE IF NOT EXISTS ml_models (
    model_id VARCHAR(255) PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    version VARCHAR(50) NOT NULL,
    file_path TEXT,
    metadata JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Configuration table
CREATE TABLE IF NOT EXISTS configurations (
    config_key VARCHAR(255) PRIMARY KEY,
    config_value JSONB NOT NULL,
    environment VARCHAR(50) DEFAULT 'production',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
'''
        
        main_schema_file = schema_dir / "main.sql"
        with open(main_schema_file, 'w') as f:
            f.write(main_schema)
        
        # Events database schema
        events_schema = '''-- SKZ Agents Events Database Schema

-- Sync transactions log
CREATE TABLE IF NOT EXISTS sync_transactions (
    transaction_id VARCHAR(255) PRIMARY KEY,
    operation VARCHAR(50) NOT NULL,
    source_system VARCHAR(100) NOT NULL,
    target_system VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    entity_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    checksum VARCHAR(64)
);

-- Create indexes
CREATE INDEX idx_sync_transactions_entity ON sync_transactions(entity_type, entity_id);
CREATE INDEX idx_sync_transactions_status ON sync_transactions(status);
CREATE INDEX idx_sync_transactions_created_at ON sync_transactions(created_at);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(255),
    user_id VARCHAR(255),
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Performance metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255) NOT NULL,
    metric_value NUMERIC NOT NULL,
    metric_unit VARCHAR(50),
    component VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
'''
        
        events_schema_file = schema_dir / "events.sql"
        with open(events_schema_file, 'w') as f:
            f.write(events_schema)
        
        print(f"🗄️ Created database schemas: {schema_dir}")
    
    def _setup_development_environment(self):
        """Setup development environment variables for testing"""
        dev_env = {
            'ENVIRONMENT': 'development',
            'DATABASE_URL': 'postgresql://skz_user:dev_password@localhost:5432/skz_dev',
            'REDIS_URL': 'redis://localhost:6379/0',
            'USPTO_API_KEY': 'development_key_placeholder',
            'GOOGLE_PATENTS_API_KEY': 'development_key_placeholder',
            'SENDGRID_API_KEY': 'development_key_placeholder',
            'BERT_MODEL_PATH': './models/bert-base-uncased',
            'JWT_SECRET_KEY': 'development_secret_key_not_for_production',
            'LOG_LEVEL': 'DEBUG'
        }
        
        # Write to .env file for development
        env_file = self.repo_path / ".env"
        with open(env_file, 'w') as f:
            f.write("# Development Environment Configuration\n")
            f.write("# DO NOT USE IN PRODUCTION\n\n")
            for key, value in dev_env.items():
                f.write(f"{key}={value}\n")
        
        print(f"🛠️ Created development environment file: {env_file}")
    
    def _print_setup_instructions(self):
        """Print setup instructions"""
        instructions = """
🎉 Production Environment Setup Complete!

NEXT STEPS:

1. 📋 Configuration Setup:
   - Copy .env.production.template to .env.production
   - Fill in all required API keys and credentials
   - Verify database connection strings

2. 🔐 Security Setup:
   - Generate JWT secret key: python -c "import secrets; print(secrets.token_hex(32))"
   - Obtain SSL certificates for HTTPS
   - Configure firewall rules

3. 🗄️ Database Setup:
   - Create PostgreSQL databases as specified in schema/
   - Run schema files to create tables
   - Set up database users and permissions

4. 🚀 Deployment Options:

   A. Docker Deployment:
      docker-compose -f docker-compose.production.yml up -d

   B. Systemd Service:
      sudo cp systemd/skz-agents.service /etc/systemd/system/
      sudo systemctl daemon-reload
      sudo systemctl enable skz-agents
      sudo systemctl start skz-agents

   C. Manual Deployment:
      python3 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      python3 src/main.py

5. 🔍 Validation:
   python3 comprehensive_production_validator.py

6. 📊 Monitoring:
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000
   - Health Check: http://localhost:5000/health

For detailed documentation, see:
- README.md
- docs/deployment.md
- docs/configuration.md
"""
        print(instructions)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Environment Setup")
    parser.add_argument('--template-only', action='store_true', help="Create templates only, no dev env")
    parser.add_argument('--validate', action='store_true', help="Validate existing configuration")
    
    args = parser.parse_args()
    
    setup = ProductionEnvironmentSetup()
    
    if args.validate:
        # Run validation
        os.system("python3 comprehensive_production_validator.py")
    else:
        setup.setup_production_environment(template_only=args.template_only)
    
    return 0

if __name__ == "__main__":
    exit(main())