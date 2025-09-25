-- SKZ Agents Main Database Schema

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
