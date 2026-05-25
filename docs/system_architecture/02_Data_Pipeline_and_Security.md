# MARKET INTELLIGENCE SYSTEM: COMPREHENSIVE ARCHITECTURE & RESEARCH WHITEPAPER

## PART 2: DATA ENGINEERING, DATABASE INFRASTRUCTURE, & CAUSAL INTEGRITY

## 1. IN-PROCESS COLUMNAR OLAP DATA WAREHOUSE (DUCKDB)
The system storage engine uses `DuckDB`, an embedded, columnar relational database management system designed specifically for `Online Analytical Processing (OLAP)`. Traditional row-based databases (such as `SQLite` or `MySQL`) store data tuple-by-tuple, which introduces significant `CPU` overhead and cache-miss latency when performing analytical scans, rolling windows, and aggregations across millions of rows.

`DuckDB` solves these bottlenecks through two core features:

### A. Columnar Storage Layout
Data is stored column-by-column rather than row-by-row. When the feature engineering pipeline requires only three features (e.g., Close price, `CPI_MoM`, and `YieldCurve_10Y2Y`) from a table containing fifty columns, `DuckDB` reads only those three columns from disk. This reduces `I/O` bottlenecking and maximizes storage efficiency.

### B. Vectorized Query Execution
Instead of processing data row-by-row (tuple-at-a-time execution), `DuckDB`'s execution engine processes data in column vectors (arrays of data points, typically 1024 values per vector). This layout fits directly into `CPU L1/L2` cache lines, maximizing compiler loop-unrolling efficiency and utilizing `SIMD (Single Instruction, Multiple Data)` processor instructions.

## 2. MULTI-THREADED RUST ENGINE INTEGRATION (POLARS)
To process the extracted database queries without Python-level `Global Interpreter Lock (GIL)` bottlenecks, the system integrates `Polars`, a Rust-based, multi-threaded `DataFrame` library. `Polars` is built on the `Apache Arrow` memory specification, enabling zero-copy memory mapping.

When a query is executed, `DuckDB` constructs a relation object. By calling the `rel.pl()` method, the data is projected directly into a `Polars DataFrame`. This operation simply maps the pointers to the existing memory buffer allocated by `Arrow`, entirely bypassing the `CPU` overhead of serializing, parsing, and copying data blocks. `Polars` then performs downstream feature calculations using lazy evaluation: it parses the query into a logical execution graph, optimizes it (e.g., combining projection and filter pushdowns), and executes the calculations in parallel across all available `CPU` cores.

## 3. PSEUDO-ACID DUAL-WRITE ARCHITECTURE AND CONCURRENCY CONTROL
Production financial systems must guarantee data persistence and `UI` responsiveness during concurrent write operations. The `MarketDataStore` class implements a dual-write architecture that models transactional integrity:

### A. Multi-User Concurrency Control
`Streamlit` pages run in separate threads, meaning a user reloading the `UI` can trigger a read operation at the exact moment the daily background scheduler attempts to write new market data. To prevent database locks, read operations establish `DuckDB` connections with `read_only=True`. This allows multiple concurrent threads to query the database simultaneously without blocking.

### B. Write Pipeline and Atomicity
Write operations are executed using a separate read-write connection. The write pipeline runs:
```sql
CREATE OR REPLACE TABLE table_name AS SELECT * FROM temp_df
```
This replaces the target table atomically. If the write succeeds, the store immediately exports the updated table as a backup `.csv` file in the `data/` directory.

### C. Fallback Redundancy and Recovery
In environments where file systems are shared or antivirus software locks the `.db` file, write operations can encounter `OS`-level file locks. The `MarketDataStore` catches these write exceptions, updates the backup `CSV` file, and logs a warning. The query pipeline is designed with a fallback cascade: if the `DuckDB` connection fails, the system automatically falls back to reading from the backup `CSV` files. This guarantees that `UI` operations never crash due to database locking. (Phase 6 roadmaps plan timestamp-based database auto-healing: upon database re-connection, it checks if the `CSV` backup modified-time is newer than the `DB` table, and automatically executes a re-migration).

## 4. CAUSAL INTEGRITY AND LOOK-AHEAD BIAS PREVENTION
Look-ahead bias occurs when a machine learning model is trained on, or evaluates features using, information that was not publicly available at the theoretical time of prediction. In backtesting, this leads to artificially inflated accuracy metrics that collapse when deployed live. The system implements three strict pipeline constraints to prevent leakage:

### A. Automated Lag Ingestion
Macroeconomic indicators (such as `CPI`, `PPI`, `PCE`, and `Non-Farm Payrolls`) are released by government agencies with varying delays (ranging from two to six weeks) and are frequently revised in subsequent months. To prevent look-ahead bias, the feature pipeline lags `FRED` monthly features using automated shifts (e.g., `CPI_MoM_lag3`, `NFP_Change_lag3`). This models the true information set available to an analyst at time $t$, ensuring the models train only on finalized, publicly accessible data.

### B. Commitment of Traders (COT) Release Date Realignment
The `CFTC` gathers `Commitment of Traders` futures positioning data as of Tuesday close. However, this report is not released to the public until Friday evening (typically 15:30 `EST`). Merging Tuesday's positioning data directly into Tuesday's daily price record would leak three days of future information. The `COTFetcher` pipeline shifts the raw data date:
$$
\text{Date}_{\text{aligned}} = \text{Date}_{\text{CFTC Tuesday}} + 3 \text{ days}
$$
This re-indexes the `COT` report to Friday, aligning the data release with the actual public market availability.

### C. Strict Forward-Filling and Interpolation Ban
Macroeconomic data is reported monthly or weekly, whereas market prices are daily. To merge these mixed-frequency datasets, the system forward-fills data: Friday's `COT` value is propagated forward through Saturday, Sunday, and the next week (`ffill()`) until a new release is detected. The pipeline strictly prohibits linear interpolation. Interpolating between a Tuesday `COT` report and the next Tuesday `COT` report would utilize future data points to calculate the values for Wednesday through Monday, introducing severe look-ahead leakage.