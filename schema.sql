-- Esquema de base de datos para Funding Fee Scalper v5 (Quant Edition)
-- Compatible con PostgreSQL / Supabase / cualquier DB PostgreSQL

CREATE TABLE IF NOT EXISTS trade_logs (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol      TEXT NOT NULL,
    side        TEXT NOT NULL,
    rule        TEXT,
    margin_usdt DOUBLE PRECISION,
    amount      DOUBLE PRECISION,
    price       DOUBLE PRECISION,
    order_id    TEXT,
    status      TEXT,
    fill_confirmed BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_trade_logs_symbol ON trade_logs(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_logs_ts ON trade_logs(ts DESC);

CREATE TABLE IF NOT EXISTS pnl_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol          TEXT NOT NULL,
    entry_price     DOUBLE PRECISION,
    mark_price      DOUBLE PRECISION,
    position_amt    DOUBLE PRECISION,
    unrealized_pnl  DOUBLE PRECISION,
    roe_pct         DOUBLE PRECISION,
    margin_used     DOUBLE PRECISION,
    funding_rate    DOUBLE PRECISION,
    funding_interval_hours DOUBLE PRECISION,
    annualized_fr_pct      DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_symbol ON pnl_snapshots(symbol);
CREATE INDEX IF NOT EXISTS idx_pnl_snapshots_ts ON pnl_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS math_snapshots (
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol              TEXT NOT NULL,

    -- Power Law / Tail Risk
    tail_alpha          DOUBLE PRECISION,
    tail_pvalue         DOUBLE PRECISION,
    tail_risk_index     DOUBLE PRECISION,
    hill_xi             DOUBLE PRECISION,

    -- EVT (Extreme Value Theory)
    evt_var_95          DOUBLE PRECISION,
    evt_var_99          DOUBLE PRECISION,
    evt_es_99           DOUBLE PRECISION,
    evt_xi              DOUBLE PRECISION,
    evt_sigma           DOUBLE PRECISION,
    evt_stop_price      DOUBLE PRECISION,

    -- Hurst / Persistencia
    hurst_fr            DOUBLE PRECISION,
    hurst_method        TEXT,
    fr_regime           TEXT,  -- persistent_negative, persistent_positive, mean_reverting, random_walk

    -- Entropía / Teoría de la Información
    shannon_entropy_fr  DOUBLE PRECISION,
    te_oi_to_fr         DOUBLE PRECISION,
    te_fr_to_oi         DOUBLE PRECISION,
    mi_fr_oi            DOUBLE PRECISION,

    -- Volatilidad Logarítmica
    log_rv_annualized   DOUBLE PRECISION,
    log_vol_regime      TEXT,  -- calm, normal, extreme
    log_atr_14          DOUBLE PRECISION,

    -- Multifractalidad
    mf_width_delta_alpha DOUBLE PRECISION,
    mf_spectrum_json    JSONB,

    -- Score integrador
    power_score         DOUBLE PRECISION,
    math_gates_passed   BOOLEAN,
    gates_blocked_by    TEXT
);

CREATE INDEX IF NOT EXISTS idx_math_snapshots_symbol ON math_snapshots(symbol);
CREATE INDEX IF NOT EXISTS idx_math_snapshots_ts ON math_snapshots(ts DESC);

CREATE TABLE IF NOT EXISTS learning_stats (
    symbol           TEXT PRIMARY KEY,
    sample_count     INTEGER NOT NULL,
    wins             INTEGER NOT NULL,
    losses           INTEGER NOT NULL,
    win_rate         DOUBLE PRECISION NOT NULL,
    avg_roe_pct      DOUBLE PRECISION NOT NULL,
    best_roe_pct     DOUBLE PRECISION NOT NULL,
    worst_roe_pct    DOUBLE PRECISION NOT NULL,
    avg_hold_minutes DOUBLE PRECISION,
    score_multiplier DOUBLE PRECISION NOT NULL,
    size_multiplier  DOUBLE PRECISION NOT NULL,
    updated_ts       TIMESTAMPTZ NOT NULL DEFAULT now()
);
