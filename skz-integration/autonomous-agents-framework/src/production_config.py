
"""
Production Configuration Management
==================================
Centralized configuration for all SKZ autonomous agents
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class ProductionConfig:
    """Production-grade configuration management"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv('SKZ_CONFIG_FILE', 'config/production.json')
        self.config = self._load_config()
        self._validate_required_settings()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment"""
        config = {}
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            with open(self.config_file) as f:
                config = json.load(f)
        
        # Override with environment variables
        env_mappings = {
            'DATABASE_URL': 'database.url',
            'REDIS_URL': 'redis.url',
            'USPTO_API_KEY': 'apis.uspto.api_key',
            'GOOGLE_PATENTS_API_KEY': 'apis.google_patents.api_key',
            'SENDGRID_API_KEY': 'communication.email.sendgrid.api_key',
            'TWILIO_API_KEY': 'communication.sms.twilio.api_key',
            'BERT_MODEL_PATH': 'ml.bert.model_path',
            'HUGGINGFACE_API_KEY': 'ml.huggingface.api_key'
        }
        
        for env_key, config_path in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value:
                self._set_nested_config(config, config_path, env_value)
        
        return config
    
    def _validate_required_settings(self):
        """Validate all required production settings are present"""
        required_settings = [
            'database.url',
            'apis.uspto.api_key',
            'apis.google_patents.api_key',
            'communication.email.sendgrid.api_key',
            'ml.bert.model_path'
        ]
        
        missing = []
        for setting in required_settings:
            if not self._get_nested_config(self.config, setting):
                missing.append(setting)
        
        if missing:
            raise ValueError(f"Missing required production settings: {', '.join(missing)}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._get_nested_config(self.config, key) or default
    
    def _get_nested_config(self, config: Dict, key: str) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split('.')
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value
    
    def _set_nested_config(self, config: Dict, key: str, value: Any):
        """Set nested configuration value using dot notation"""  
        keys = key.split('.')
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

# Global configuration instance
production_config = ProductionConfig()
