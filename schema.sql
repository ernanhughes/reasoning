CREATE TABLE IF NOT EXISTS prompt_log (
    prompt_id TEXT PRIMARY KEY,
    model TEXT,
    dataset TEXT,
    text TEXT,
    tokens JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS activation_log (
    sae_config TEXT,
    activations_file TEXT,
    skipped BOOLEAN,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sae_training_log (
    sae_config TEXT,
    activations_file TEXT,
    final_loss FLOAT,
    epochs INT,
    created_at TIMESTAMPTZ DEFAULT now(),
    model_summary JSONB
);


CREATE TABLE IF NOT EXISTS activation_logs (
    id SERIAL PRIMARY KEY,
    sae_config TEXT,
    activations_file TEXT,
    skipped BOOLEAN,
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS activations (
    id SERIAL PRIMARY KEY,
    activation_id TEXT,
    activation_date TIMESTAMP,
    activation_status TEXT,
    activation_log_id INTEGER REFERENCES activation_logs(id)
);

CREATE TABLE IF NOT EXISTS feature_log (
    id SERIAL PRIMARY KEY,
    prompt_id TEXT REFERENCES prompt_log(prompt_id) ON DELETE CASCADE,    
    feature_id INT,
    reason_score FLOAT,
    topk_score FLOAT,
    sae_config TEXT,
    layer_index INT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS feature_vector_log (
    id SERIAL PRIMARY KEY,
    prompt_id TEXT NOT NULL,
    reason_scores JSONB NOT NULL,
    topk_scores JSONB NOT NULL,
    sae_config TEXT NOT NULL,
    layer_index INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
