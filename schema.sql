
CREATE TABLE activation_logs (
    id SERIAL PRIMARY KEY,
    sae_config TEXT,
    activations_file TEXT,
    skipped BOOLEAN,
    created_at TIMESTAMP
);

CREATE TABLE activations (
    id SERIAL PRIMARY KEY,
    activation_id TEXT,
    activation_date TIMESTAMP,
    activation_status TEXT,
    activation_log_id INTEGER REFERENCES activation_logs(id)
);


