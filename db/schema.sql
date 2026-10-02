CREATE TABLE IF NOT EXISTS house_prices (
    id BIGSERIAL PRIMARY KEY,
    address TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    bedrooms INTEGER NOT NULL,
    bathrooms NUMERIC(4,1) NOT NULL,
    area_sqft INTEGER NOT NULL,
    year_built INTEGER NOT NULL,
    price NUMERIC(14,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_house_prices_city ON house_prices(city);
CREATE INDEX IF NOT EXISTS idx_house_prices_bedrooms ON house_prices(bedrooms);
CREATE INDEX IF NOT EXISTS idx_house_prices_price ON house_prices(price);
CREATE INDEX IF NOT EXISTS idx_house_prices_year_built ON house_prices(year_built);

CREATE TABLE IF NOT EXISTS conversations (
    id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id BIGSERIAL PRIMARY KEY,
    conversation_id VARCHAR(100) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

DO $$
BEGIN
    IF (SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'conversations' AND column_name = 'id') = 'uuid' THEN
        ALTER TABLE messages DROP CONSTRAINT IF EXISTS messages_conversation_id_fkey;
        ALTER TABLE conversations ALTER COLUMN id TYPE VARCHAR(100) USING id::text;
        ALTER TABLE messages ALTER COLUMN conversation_id TYPE VARCHAR(100) USING conversation_id::text;
        ALTER TABLE messages ADD CONSTRAINT messages_conversation_id_fkey
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE;
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS long_term_memories (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
