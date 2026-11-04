-- One row per saved inspection, refreshed from the source SQLite database.
CREATE TABLE IF NOT EXISTS bi_records (
    created_at timestamp NOT NULL,
    asset_type text NOT NULL,
    predicted_class text NOT NULL,
    state_key text NOT NULL,
    work_status text NOT NULL,
    confidence double precision NOT NULL,
    input_type text NOT NULL
);

CREATE OR REPLACE VIEW bi_inspection_total AS
SELECT COUNT(*) AS inspection_count FROM bi_records;

-- Recreate these derived views atomically when their grouping columns change.
DROP VIEW IF EXISTS bi_diagnoses;
CREATE VIEW bi_diagnoses AS
SELECT predicted_class, input_type, COUNT(*) AS inspection_count
FROM bi_records GROUP BY predicted_class, input_type;

DROP VIEW IF EXISTS bi_work_status;
CREATE VIEW bi_work_status AS
SELECT work_status, input_type, COUNT(*) AS inspection_count
FROM bi_records GROUP BY work_status, input_type;

CREATE OR REPLACE VIEW bi_inspection_details AS
SELECT CAST(created_at AS date) AS inspection_date, asset_type, predicted_class,
       state_key, work_status, input_type, COUNT(*) AS inspection_count,
       AVG(confidence) AS avg_confidence
FROM bi_records
GROUP BY CAST(created_at AS date), asset_type, predicted_class, state_key, work_status, input_type;
