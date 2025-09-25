-- SKZ Agents Events Database Schema

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
