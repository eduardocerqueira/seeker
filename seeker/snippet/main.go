//date: 2025-12-23T17:08:44Z
//url: https://api.github.com/gists/5e984521217360afa3f577fc8b5db779
//owner: https://api.github.com/users/ramanguleria

package main

import (
	"bytes"
	"database/sql"
	"fmt"
	"io"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "github.com/lib/pq"
)

// ==============================================================================
// CONFIGURATION
// ==============================================================================
const (
	DBHost         = "10.173.64.19" // PostgreSQL Instance IP
	DBPort         = 5432
	DBUser         = "postgres"
	DBName         = "tpcc"
	NumWorkers     = 20               // Concurrency Level (reduced to avoid I/O saturation)
	ChunkSizeBytes = 10 * 1024 * 1024 // 10MB chunks (legacy, now adaptive)
	// Password should be set via PGPASSWORD environment variable

	// Adaptive chunking configuration
	// NOTE: Larger chunks reduce I/O operations and disk contention
	MinChunkSizeBytes     = 50 * 1024 * 1024          // 50MB minimum (reduced I/O ops)
	MaxChunkSizeBytes     = 1024 * 1024 * 1024        // 1GB maximum (larger sequential reads)
	TargetChunksPerWorker = 5                         // Fewer chunks per worker (less I/O contention)
	TinyTableThreshold    = 100 * 1024 * 1024         // 100MB - very small tables
	SmallTableThreshold   = 1 * 1024 * 1024 * 1024    // 1GB - single COPY threshold (balance IOPS vs parallelism)
	LargeTableThreshold   = 100 * 1024 * 1024 * 1024  // 100GB - very large tables
	HugeTableThreshold    = 1024 * 1024 * 1024 * 1024 // 1TB - massive tables

	// Performance model constants (MB/s)
	SequentialScanThroughput = 200.0 // Single COPY sequential scan
	ParallelScanThroughput   = 250.0 // Parallel COPY with chunking
	ChunkCoordinationCostMs  = 5.0   // Overhead per chunk in milliseconds
)

// ==============================================================================
// DATA STRUCTURES
// ==============================================================================

type DumpStrategy int

const (
	StrategyCopySingleTiny  DumpStrategy = iota // < 100MB, single COPY (no ORDER BY, IOPS-light, fast)
	StrategyCopySingleSmall                     // 100MB-1GB, single COPY (no ORDER BY, IOPS-light, sequential)
	StrategyCopySingleNoPK                      // Any size, no suitable PK (single COPY, no ORDER BY)
	StrategyCopyChunkMedium                     // 1GB-100GB, chunked COPY with ORDER BY (parallel, dedup-friendly)
	StrategyCopyChunkLarge                      // > 100GB, chunked COPY with ORDER BY (parallel, dedup-friendly)
)

func (s DumpStrategy) String() string {
	switch s {
	case StrategyCopySingleTiny:
		return "Copy-Single-Tiny"
	case StrategyCopySingleSmall:
		return "Copy-Single-Small"
	case StrategyCopySingleNoPK:
		return "Copy-Single-NoPK"
	case StrategyCopyChunkMedium:
		return "Copy-Chunk-Medium"
	case StrategyCopyChunkLarge:
		return "Copy-Chunk-Large"
	default:
		return "Unknown"
	}
}

type TableInfo struct {
	Schema         string
	Name           string
	FullName       string
	NumRows        int64
	SizeBytes      int64
	AvgRowLen      int64
	PrimaryKeyCols []string
	PKColumnTypes  []string
	Strategy       DumpStrategy // Selected dump strategy
	ChunkSize      int64        // Adaptive chunk size for this table
}

type ChunkTask struct {
	Table       *TableInfo
	ChunkID     int
	WhereClause string
	Strategy    DumpStrategy
	Priority    int // For scheduling (lower = higher priority)
}

type MinMaxResult struct {
	Table *TableInfo
	Min   int64
	Max   int64
	Err   error
}

// ==============================================================================
// DATABASE CONNECTION
// ==============================================================================

func openDB() (*sql.DB, error) {
	// Password is read from PGPASSWORD environment variable
	connStr := fmt.Sprintf("host=%s port=%d user=%s dbname=%s sslmode=require",
		DBHost, DBPort, DBUser, DBName)

	db, err := sql.Open("postgres", connStr)
	if err != nil {
		return nil, err
	}

	if err := db.Ping(); err != nil {
		return nil, err
	}

	db.SetMaxOpenConns(NumWorkers * 2)
	db.SetMaxIdleConns(NumWorkers)

	return db, nil
}

// ==============================================================================
// TABLE DISCOVERY
// ==============================================================================

func discoverTables(db *sql.DB) ([]*TableInfo, error) {

	query := `
		SELECT
			n.nspname AS schema,
			c.relname AS table_name,
			pg_total_relation_size(c.oid) AS total_size,
			CASE
				WHEN c.reltuples > 0 THEN c.reltuples::bigint
				WHEN c.relpages > 0 THEN (c.relpages * 100)::bigint
				ELSE 1000
			END AS estimated_rows
		FROM pg_class c
		JOIN pg_namespace n ON n.oid = c.relnamespace
		WHERE c.relkind = 'r'
		AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
		ORDER BY total_size DESC
	`

	rows, err := db.Query(query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var tables []*TableInfo
	for rows.Next() {
		var t TableInfo
		var estimatedRows int64
		if err := rows.Scan(&t.Schema, &t.Name, &t.SizeBytes, &estimatedRows); err != nil {
			return nil, err
		}
		t.FullName = fmt.Sprintf("%s.%s", t.Schema, t.Name)

		// Skip truly empty tables (0 bytes)
		if t.SizeBytes == 0 {
			continue
		}

		t.NumRows = estimatedRows

		// Calculate average row length
		if t.SizeBytes > 0 && t.NumRows > 0 {
			t.AvgRowLen = t.SizeBytes / t.NumRows
		} else {
			t.AvgRowLen = 256
		}

		// Ensure minimum values
		if t.AvgRowLen == 0 {
			t.AvgRowLen = 256
		}
		if t.NumRows == 0 {
			t.NumRows = 1000
		}

		tables = append(tables, &t)
	}

	return tables, nil
}

// ==============================================================================
// PRIMARY KEY DISCOVERY
// ==============================================================================

func fetchPrimaryKeys(db *sql.DB, table *TableInfo) error {
	query := `
		SELECT a.attname, t.typname
		FROM pg_index i
		JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
		JOIN pg_type t ON t.oid = a.atttypid
		WHERE i.indrelid = $1::regclass
		AND i.indisprimary
		ORDER BY array_position(i.indkey, a.attnum)
	`

	rows, err := db.Query(query, table.FullName)
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var colName, colType string
		if err := rows.Scan(&colName, &colType); err != nil {
			return err
		}
		table.PrimaryKeyCols = append(table.PrimaryKeyCols, colName)
		table.PKColumnTypes = append(table.PKColumnTypes, colType)
	}

	return nil
}

// ==============================================================================
// PARALLEL MIN/MAX QUERIES
// ==============================================================================

func parallelMinMax(tables []*TableInfo, db *sql.DB) map[string]MinMaxResult {
	results := make(chan MinMaxResult, len(tables))
	var wg sync.WaitGroup

	for _, table := range tables {
		// Only fetch MIN/MAX for tables that will be chunked
		if table.Strategy != StrategyCopyChunkMedium && table.Strategy != StrategyCopyChunkLarge {
			continue
		}

		if len(table.PrimaryKeyCols) == 0 || !isIntegerType(table.PKColumnTypes[0]) {
			continue
		}

		wg.Add(1)
		go func(t *TableInfo) {
			defer wg.Done()

			pkCol := t.PrimaryKeyCols[0]
			query := fmt.Sprintf("SELECT MIN(\"%s\"), MAX(\"%s\") FROM %s", pkCol, pkCol, t.FullName)

			var min, max int64
			err := db.QueryRow(query).Scan(&min, &max)

			results <- MinMaxResult{
				Table: t,
				Min:   min,
				Max:   max,
				Err:   err,
			}
		}(table)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	// Collect results
	resultMap := make(map[string]MinMaxResult)
	for r := range results {
		resultMap[r.Table.FullName] = r
	}

	return resultMap
}

// ==============================================================================
// INTERLEAVED TASK SCHEDULING
// ==============================================================================

func interleaveTasks(tasks []ChunkTask) []ChunkTask {
	// Group tasks by strategy
	buckets := make(map[DumpStrategy][]ChunkTask)
	for _, task := range tasks {
		buckets[task.Strategy] = append(buckets[task.Strategy], task)
	}

	// Weighted round-robin: take N tasks from each bucket in proportion to importance
	weights := map[DumpStrategy]int{
		StrategyCopyChunkLarge:  100, // Large chunks: add many at once
		StrategyCopyChunkMedium: 20,  // Medium chunks: add some
		StrategyCopySingleSmall: 5,   // Small tables: add a few
		StrategyCopySingleTiny:  10,  // Tiny tables: add frequently to avoid starvation
		StrategyCopySingleNoPK:  5,   // No PK tables: add a few
	}

	var result []ChunkTask
	anyBucketHasTasks := true

	for anyBucketHasTasks {
		anyBucketHasTasks = false

		for strategy, weight := range weights {
			bucket := buckets[strategy]
			if len(bucket) == 0 {
				continue
			}

			anyBucketHasTasks = true

			// Take up to 'weight' tasks from this bucket
			takeCount := weight
			if takeCount > len(bucket) {
				takeCount = len(bucket)
			}

			result = append(result, bucket[:takeCount]...)
			buckets[strategy] = bucket[takeCount:]
		}
	}

	return result
}

// ==============================================================================
// STRATEGY SELECTION & ADAPTIVE CHUNKING
// ==============================================================================

func hasSuitablePK(table *TableInfo) bool {
	if len(table.PrimaryKeyCols) == 0 {
		return false
	}

	// Single-column integer PK is ideal
	if len(table.PrimaryKeyCols) == 1 && isIntegerType(table.PKColumnTypes[0]) {
		return true
	}

	// Multi-column PK: only suitable if first column is integer
	if isIntegerType(table.PKColumnTypes[0]) {
		return true
	}

	// String/UUID PKs: can use LIMIT/OFFSET but less efficient
	return true
}

func estimateMinMaxTime(sizeBytes int64) float64 {
	// MIN/MAX on indexed PK is fast (index scan)
	const GB = 1024 * 1024 * 1024
	if sizeBytes < GB {
		return 0.1 // 100ms
	} else if sizeBytes < 100*GB {
		return 0.5 // 500ms
	} else {
		return 2.0 // 2 seconds for very large tables
	}
}

func calculateOptimalChunkSize(table *TableInfo, numWorkers int) int64 {
	// Calculate ideal chunk size to keep workers busy
	targetChunks := int64(numWorkers * TargetChunksPerWorker)
	idealChunkSize := table.SizeBytes / targetChunks

	// Clamp to reasonable bounds
	if idealChunkSize < MinChunkSizeBytes {
		return MinChunkSizeBytes
	}
	if idealChunkSize > MaxChunkSizeBytes {
		return MaxChunkSizeBytes
	}

	return idealChunkSize
}

func selectDumpStrategy(table *TableInfo, numWorkers int) DumpStrategy {
	// 1. Check size-based fast paths
	if table.SizeBytes < TinyTableThreshold {
		return StrategyCopySingleTiny
	}

	// 2. Check if table has suitable PK for chunking
	if !hasSuitablePK(table) {
		return StrategyCopySingleNoPK
	}

	// 3. For small tables, use single COPY (chunking overhead not worth it)
	if table.SizeBytes < SmallTableThreshold {
		return StrategyCopySingleSmall
	}

	// 4. Calculate costs for both strategies
	const MB = 1024 * 1024

	// Single COPY cost (single-threaded, sequential scan)
	singleCopyTimeSec := float64(table.SizeBytes/MB) / SequentialScanThroughput

	// Chunked COPY cost (parallel)
	optimalChunkSize := calculateOptimalChunkSize(table, numWorkers)
	numChunks := table.SizeBytes / optimalChunkSize
	if numChunks == 0 {
		numChunks = 1
	}

	// Overhead: MIN/MAX query + chunk coordination
	chunkingOverhead := estimateMinMaxTime(table.SizeBytes) +
		(float64(numChunks) * ChunkCoordinationCostMs / 1000.0)

	// Parallel dump time (assuming perfect scaling up to numWorkers)
	parallelWorkers := int64(numWorkers)
	if numChunks < parallelWorkers {
		parallelWorkers = numChunks
	}

	chunkingTimeSec := chunkingOverhead +
		(float64(table.SizeBytes/MB) / (ParallelScanThroughput * float64(parallelWorkers)))

	// 5. Choose strategy with lower cost
	if singleCopyTimeSec < chunkingTimeSec {
		return StrategyCopySingleSmall
	}

	// 6. Classify by size
	if table.SizeBytes >= LargeTableThreshold {
		return StrategyCopyChunkLarge
	}

	return StrategyCopyChunkMedium
}

// ==============================================================================
// CHUNKING LOGIC
// ==============================================================================

type Splitter struct {
	table        *TableInfo
	rowCount     int64
	rowsPerChunk int64
	accuracy     int64
	db           *sql.DB
	chunks       []ChunkTask
}

func NewSplitter(table *TableInfo, chunkSizeBytes int64, db *sql.DB) *Splitter {
	rowsPerChunk := chunkSizeBytes / table.AvgRowLen
	if rowsPerChunk == 0 {
		rowsPerChunk = 1000
	}

	accuracy := max(rowsPerChunk/10, 10)

	return &Splitter{
		table:        table,
		rowCount:     table.NumRows,
		rowsPerChunk: rowsPerChunk,
		accuracy:     accuracy,
		db:           db,
		chunks:       make([]ChunkTask, 0),
	}
}

func max(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b int64) int64 {
	if a < b {
		return a
	}
	return b
}

func abs(a int64) int64 {
	if a < 0 {
		return -a
	}
	return a
}

func isIntegerType(typeName string) bool {
	switch typeName {
	case "int2", "int4", "int8", "smallint", "integer", "bigint",
		"smallserial", "serial", "bigserial":
		return true
	default:
		return false
	}
}

// Simple linear chunking - no COUNT queries needed
// Divides the ID range evenly based on estimated row count

// Split table into chunks for integer primary keys using simple linear division
// No COUNT queries - just divides ID range evenly
// minVal and maxVal should be pre-fetched via parallelMinMax
func (s *Splitter) chunkIntegerColumnWithMinMax(minVal, maxVal int64) error {
	if len(s.table.PrimaryKeyCols) == 0 {
		return fmt.Errorf("no primary key found")
	}

	pkCol := s.table.PrimaryKeyCols[0]

	if minVal == maxVal {
		// Single row or all same value
		whereClause := fmt.Sprintf("\"%s\" = %d", pkCol, minVal)
		s.chunks = append(s.chunks, ChunkTask{
			Table:       s.table,
			ChunkID:     0,
			WhereClause: whereClause,
			Strategy:    s.table.Strategy,
		})
		return nil
	}

	// Calculate number of chunks needed based on adaptive chunk size
	estimatedChunks := s.table.SizeBytes / s.table.ChunkSize
	if estimatedChunks == 0 {
		estimatedChunks = 1
	}

	// Simple linear division of ID range
	idRange := maxVal - minVal
	chunkIDSize := idRange / estimatedChunks
	if chunkIDSize == 0 {
		chunkIDSize = 1
	}

	currentID := minVal
	chunkID := 0

	for currentID <= maxVal {
		beginID := currentID
		endID := currentID + chunkIDSize

		// Last chunk gets everything remaining
		if endID > maxVal || chunkID == int(estimatedChunks)-1 {
			endID = maxVal
		}

		whereClause := fmt.Sprintf("\"%s\" >= %d AND \"%s\" <= %d",
			pkCol, beginID, pkCol, endID)

		s.chunks = append(s.chunks, ChunkTask{
			Table:       s.table,
			ChunkID:     chunkID,
			WhereClause: whereClause,
			Strategy:    s.table.Strategy,
		})

		chunkID++
		currentID = endID + 1

		// Break if we've covered the range
		if endID >= maxVal {
			break
		}
	}

	return nil
}

// Split table into chunks for non-integer primary keys
func (s *Splitter) chunkNonIntegerColumn() error {
	if len(s.table.PrimaryKeyCols) == 0 {
		return fmt.Errorf("no primary key found")
	}

	pkCol := s.table.PrimaryKeyCols[0]
	orderBy := fmt.Sprintf("ORDER BY \"%s\"", pkCol)

	chunkID := 0
	offset := int64(0)

	// Calculate rows per chunk based on adaptive chunk size
	rowsPerChunk := s.table.ChunkSize / s.table.AvgRowLen
	if rowsPerChunk == 0 {
		rowsPerChunk = 1000
	}

	for offset < s.rowCount {
		limit := rowsPerChunk
		whereClause := fmt.Sprintf("TRUE %s LIMIT %d OFFSET %d", orderBy, limit, offset)

		s.chunks = append(s.chunks, ChunkTask{
			Table:       s.table,
			ChunkID:     chunkID,
			WhereClause: whereClause,
			Strategy:    s.table.Strategy,
		})

		chunkID++
		offset += limit
	}

	return nil
}

// Main split function - now requires pre-fetched MIN/MAX for integer PKs
func (s *Splitter) SplitWithMinMax(minVal, maxVal int64) ([]ChunkTask, error) {
	if len(s.table.PrimaryKeyCols) == 0 {
		// No primary key - dump entire table
		s.chunks = append(s.chunks, ChunkTask{
			Table:       s.table,
			ChunkID:     0,
			WhereClause: "TRUE",
			Strategy:    s.table.Strategy,
		})
		return s.chunks, nil
	}

	// Check if first PK column is integer type
	if isIntegerType(s.table.PKColumnTypes[0]) {
		if err := s.chunkIntegerColumnWithMinMax(minVal, maxVal); err != nil {
			return nil, err
		}
	} else {
		if err := s.chunkNonIntegerColumn(); err != nil {
			return nil, err
		}
	}

	return s.chunks, nil
}

// ==============================================================================
// DUMP EXECUTION
// ==============================================================================

func dumpTableWithSingleCopy(table *TableInfo, workerID int) (int64, error) {
	// Use single COPY for entire table
	// For small tables, we skip ORDER BY for performance (overhead not worth it)
	// Only medium/large chunked tables use ORDER BY for dedup consistency
	var query string

	// Simple COPY without ORDER BY (faster, less overhead)
	query = fmt.Sprintf("COPY %s TO STDOUT", table.FullName)

	// Password is read from PGPASSWORD environment variable
	cmd := exec.Command("psql",
		"-h", DBHost,
		"-p", strconv.Itoa(DBPort),
		"-U", DBUser,
		"-d", DBName,
		"-c", query,
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}

	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		return 0, err
	}

	// Dump to /dev/null and count bytes
	written, _ := io.Copy(io.Discard, stdout)

	if err := cmd.Wait(); err != nil {
		if stderr.Len() > 0 {
			return written, fmt.Errorf("COPY error: %s", stderr.String())
		}
		return written, err
	}

	return written, nil
}

func dumpChunkWithCopy(chunk ChunkTask, workerID int) (int64, error) {
	// Build COPY command with WHERE clause and ORDER BY for deterministic output
	var query string
	if len(chunk.Table.PrimaryKeyCols) > 0 {
		// Order by PK for consistent output (critical for dedup)
		pkCols := strings.Join(chunk.Table.PrimaryKeyCols, ", ")
		query = fmt.Sprintf("COPY (SELECT * FROM %s WHERE %s ORDER BY %s) TO STDOUT",
			chunk.Table.FullName, chunk.WhereClause, pkCols)
	} else {
		// No PK: no ORDER BY (shouldn't happen for chunked tables)
		query = fmt.Sprintf("COPY (SELECT * FROM %s WHERE %s) TO STDOUT",
			chunk.Table.FullName, chunk.WhereClause)
	}

	// Password is read from PGPASSWORD environment variable
	cmd := exec.Command("psql",
		"-h", DBHost,
		"-p", strconv.Itoa(DBPort),
		"-U", DBUser,
		"-d", DBName,
		"-c", query,
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}

	stderr := &bytes.Buffer{}
	cmd.Stderr = stderr

	if err := cmd.Start(); err != nil {
		return 0, err
	}

	// Dump to /dev/null and count bytes
	written, _ := io.Copy(io.Discard, stdout)

	if err := cmd.Wait(); err != nil {
		if stderr.Len() > 0 {
			return written, fmt.Errorf("psql error: %s", stderr.String())
		}
		return written, err
	}

	return written, nil
}

func dumpTask(task ChunkTask, workerID int) (int64, error) {
	// Route to appropriate dump method based on strategy
	switch task.Strategy {
	case StrategyCopySingleTiny, StrategyCopySingleSmall, StrategyCopySingleNoPK:
		return dumpTableWithSingleCopy(task.Table, workerID)
	case StrategyCopyChunkMedium, StrategyCopyChunkLarge:
		return dumpChunkWithCopy(task, workerID)
	default:
		return 0, fmt.Errorf("unknown strategy: %v", task.Strategy)
	}
}

// ==============================================================================
// MAIN EXECUTION
// ==============================================================================

func main() {
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("[%s] POSTGRESQL CHUNKED PARALLEL BACKUP\n", timestamp())
	fmt.Printf("[%s] Target: %s:%d | DB: %s\n", timestamp(), DBHost, DBPort, DBName)
	fmt.Printf("[%s] Concurrency: %d Workers | Chunk Size: %s\n",
		timestamp(), NumWorkers, formatBytes(ChunkSizeBytes))
	fmt.Println(strings.Repeat("=", 70))

	overallStart := time.Now()

	// 1. Connect to database
	logln("Phase 1: Connecting to database...")
	connectStart := time.Now()
	db, err := openDB()
	if err != nil {
		log.Fatalf("[%s] FATAL: Could not connect to database: %v", timestamp(), err)
	}
	defer db.Close()
	connectDuration := time.Since(connectStart)
	logf("✅ Connected in %.2f seconds", connectDuration.Seconds())

	// 2. Discover tables
	logln("Phase 2: Discovering tables...")
	discoveryStart := time.Now()
	tables, err := discoverTables(db)
	if err != nil {
		log.Fatalf("[%s] FATAL: Could not discover tables: %v", timestamp(), err)
	}

	if len(tables) == 0 {
		log.Fatalf("[%s] FATAL: No tables found!", timestamp())
	}

	logf("✅ Found %d tables in %.2f seconds", len(tables), time.Since(discoveryStart).Seconds())
	discoveryDuration := time.Since(discoveryStart)

	// 3. Fetch primary keys for all tables
	logln("Phase 3: Fetching primary key information...")
	pkStart := time.Now()
	for _, table := range tables {
		if err := fetchPrimaryKeys(db, table); err != nil {
			log.Printf("[%s] Warning: Could not fetch PK for %s: %v", timestamp(), table.FullName, err)
		}
	}
	pkDuration := time.Since(pkStart)
	logf("✅ Fetched primary keys in %.2f seconds", pkDuration.Seconds())

	// 4. Select dump strategy for each table
	logln("Phase 4: Selecting dump strategies...")
	strategyStart := time.Now()

	strategyCount := make(map[DumpStrategy]int)
	for _, table := range tables {
		table.Strategy = selectDumpStrategy(table, NumWorkers)
		table.ChunkSize = calculateOptimalChunkSize(table, NumWorkers)
		strategyCount[table.Strategy]++
	}

	logf("Strategy selection:")
	for strategy, count := range strategyCount {
		logf("   %s: %d tables", strategy.String(), count)
	}
	strategyDuration := time.Since(strategyStart)

	// 5. Parallel MIN/MAX queries for tables that will be chunked
	logln("Phase 5: Fetching MIN/MAX values (parallel)...")
	minMaxStart := time.Now()
	minMaxResults := parallelMinMax(tables, db)
	logf("✅ Fetched MIN/MAX for %d tables in %.2f seconds", len(minMaxResults), time.Since(minMaxStart).Seconds())
	minMaxDuration := time.Since(minMaxStart)

	// 6. Generate tasks for all tables
	logln("Phase 6: Generating tasks...")
	taskGenStart := time.Now()
	var allTasks []ChunkTask
	totalTables := 0

	for _, table := range tables {
		totalTables++

		switch table.Strategy {
		case StrategyCopySingleTiny, StrategyCopySingleSmall, StrategyCopySingleNoPK:
			// Single task for entire table using single COPY
			allTasks = append(allTasks, ChunkTask{
				Table:       table,
				ChunkID:     0,
				WhereClause: "TRUE",
				Strategy:    table.Strategy,
			})

		case StrategyCopyChunkMedium, StrategyCopyChunkLarge:
			// Multiple chunks using COPY
			splitter := NewSplitter(table, table.ChunkSize, db)

			var chunks []ChunkTask
			var err error

			// Use pre-fetched MIN/MAX if available
			if minMaxResult, ok := minMaxResults[table.FullName]; ok && minMaxResult.Err == nil {
				chunks, err = splitter.SplitWithMinMax(minMaxResult.Min, minMaxResult.Max)
			} else {
				// Fallback: dump entire table
				chunks = []ChunkTask{{
					Table:       table,
					ChunkID:     0,
					WhereClause: "TRUE",
					Strategy:    StrategyCopySingleNoPK,
				}}
			}

			if err != nil {
				log.Printf("[%s] Warning: Could not split table %s: %v", timestamp(), table.FullName, err)
				// Fallback: dump entire table
				chunks = []ChunkTask{{
					Table:       table,
					ChunkID:     0,
					WhereClause: "TRUE",
					Strategy:    StrategyCopySingleNoPK,
				}}
			}

			allTasks = append(allTasks, chunks...)

			if len(chunks) > 1 {
				logf("   %s: %d chunks (%s, chunk size: %s)",
					table.FullName, len(chunks), formatBytes(table.SizeBytes), formatBytes(table.ChunkSize))
			}
		}
	}

	logf("✅ Generated %d tasks from %d tables in %.2f seconds", len(allTasks), totalTables, time.Since(taskGenStart).Seconds())
	taskGenDuration := time.Since(taskGenStart)

	// 7. Interleave tasks for optimal scheduling
	logln("Phase 7: Interleaving tasks for optimal worker utilization...")
	interleaveStart := time.Now()
	allTasks = interleaveTasks(allTasks)
	logf("✅ Interleaved %d tasks in %.2f seconds", len(allTasks), time.Since(interleaveStart).Seconds())
	interleaveDuration := time.Since(interleaveStart)

	preparationDuration := time.Since(overallStart)

	// 8. Setup worker pool
	jobs := make(chan ChunkTask, len(allTasks))
	var wg sync.WaitGroup
	var totalBytes int64
	var finishedCount int32

	for _, task := range allTasks {
		jobs <- task
	}
	close(jobs)

	logf("Phase 8: Starting %d workers for parallel dump...", NumWorkers)
	startTime := time.Now()

	// Start progress reporter
	progressDone := make(chan bool)
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				done := atomic.LoadInt32(&finishedCount)
				bytes := atomic.LoadInt64(&totalBytes)
				elapsed := time.Since(startTime).Seconds()

				// Calculate progress
				progress := float64(done) / float64(len(allTasks)) * 100
				throughput := float64(bytes) / (1024 * 1024) / elapsed

				// Estimate time remaining
				if done > 0 {
					avgTimePerTask := elapsed / float64(done)
					remainingTasks := len(allTasks) - int(done)
					etaSeconds := avgTimePerTask * float64(remainingTasks)

					// Progress bar
					barWidth := 40
					filled := int(progress / 100 * float64(barWidth))
					bar := strings.Repeat("█", filled) + strings.Repeat("░", barWidth-filled)

					fmt.Printf("\r[%s] Progress: [%s] %.1f%% | %d/%d tasks | %.1f MB/s | ETA: %s     ",
						timestamp(), bar, progress, done, len(allTasks), throughput, formatDuration(etaSeconds))
				}
			case <-progressDone:
				return
			}
		}
	}()

	// 9. Execute parallel dump
	for w := 0; w < NumWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range jobs {
				written, err := dumpTask(task, workerID)
				atomic.AddInt64(&totalBytes, written)

				if err != nil {
					log.Printf("\n[%s] [Worker %d] ❌ Failed %s chunk %d: %v",
						timestamp(), workerID, task.Table.FullName, task.ChunkID, err)
				}

				atomic.AddInt32(&finishedCount, 1)
				if written > 1024*1024 { // Log tasks > 1MB
					fmt.Printf("\n[%s] [Worker %d] ✓ %s [%s] chunk %d (%s)",
						timestamp(), workerID, task.Table.FullName, task.Strategy.String(), task.ChunkID, formatBytes(written))
				}
			}
		}(w)
	}

	wg.Wait()
	close(progressDone)
	fmt.Println() // New line after progress bar
	dumpDuration := time.Since(startTime)
	totalDuration := time.Since(overallStart)
	logf("✅ Dump completed in %.2f seconds", dumpDuration.Seconds())

	// 10. Report results
	megaBytes := float64(totalBytes) / (1024 * 1024)
	throughput := megaBytes / dumpDuration.Seconds()

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Printf("[%s] BACKUP RESULTS\n", timestamp())
	fmt.Println(strings.Repeat("=", 70))
	fmt.Printf("[%s] Total Tables     : %d\n", timestamp(), totalTables)
	fmt.Printf("[%s] Total Tasks      : %d\n", timestamp(), len(allTasks))
	fmt.Printf("[%s] Total Data       : %s\n", timestamp(), formatBytes(totalBytes))
	fmt.Printf("[%s] Throughput       : %.2f MB/s\n", timestamp(), throughput)
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("[%s] STRATEGY BREAKDOWN\n", timestamp())
	fmt.Println(strings.Repeat("-", 70))
	for strategy, count := range strategyCount {
		fmt.Printf("[%s]   %s: %d tables\n", timestamp(), strategy.String(), count)
	}
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("[%s] PHASE BREAKDOWN\n", timestamp())
	fmt.Println(strings.Repeat("-", 70))
	fmt.Printf("[%s] 1. Connection    : %.2f seconds\n", timestamp(), connectDuration.Seconds())
	fmt.Printf("[%s] 2. Discovery     : %.2f seconds\n", timestamp(), discoveryDuration.Seconds())
	fmt.Printf("[%s] 3. Primary Keys  : %.2f seconds\n", timestamp(), pkDuration.Seconds())
	fmt.Printf("[%s] 4. Strategy Sel. : %.2f seconds\n", timestamp(), strategyDuration.Seconds())
	fmt.Printf("[%s] 5. MIN/MAX (||)  : %.2f seconds\n", timestamp(), minMaxDuration.Seconds())
	fmt.Printf("[%s] 6. Task Gen.     : %.2f seconds\n", timestamp(), taskGenDuration.Seconds())
	fmt.Printf("[%s] 7. Interleaving  : %.2f seconds\n", timestamp(), interleaveDuration.Seconds())
	fmt.Printf("[%s]    Preparation   : %.2f seconds (total)\n", timestamp(), preparationDuration.Seconds())
	fmt.Printf("[%s] 8. Dump Execution: %.2f seconds\n", timestamp(), dumpDuration.Seconds())
	fmt.Printf("[%s]    TOTAL TIME    : %.2f seconds\n", timestamp(), totalDuration.Seconds())
	fmt.Println(strings.Repeat("=", 70))
}

// ==============================================================================
// UTILITIES
// ==============================================================================

func formatBytes(b int64) string {
	const unit = 1024
	if b < unit {
		return fmt.Sprintf("%d B", b)
	}
	div, exp := int64(unit), 0
	for n := b / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(b)/float64(div), "KMGTPE"[exp])
}

func timestamp() string {
	return time.Now().Format("2006-01-02 15:04:05")
}

func logf(format string, args ...interface{}) {
	fmt.Printf("[%s] %s\n", timestamp(), fmt.Sprintf(format, args...))
}

func logln(msg string) {
	fmt.Printf("[%s] %s\n", timestamp(), msg)
}

func formatDuration(seconds float64) string {
	if seconds < 60 {
		return fmt.Sprintf("%.0fs", seconds)
	}
	minutes := int(seconds / 60)
	secs := int(seconds) % 60
	if minutes < 60 {
		return fmt.Sprintf("%dm%ds", minutes, secs)
	}
	hours := minutes / 60
	minutes = minutes % 60
	return fmt.Sprintf("%dh%dm%ds", hours, minutes, secs)
}
