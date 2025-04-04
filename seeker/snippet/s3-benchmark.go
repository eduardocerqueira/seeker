//date: 2025-04-04T16:56:29Z
//url: https://api.github.com/gists/73f5ca1f777668d2c055233a0a3fe6f1
//owner: https://api.github.com/users/glommer

package main

import (
        "bytes"
        "context"
        "encoding/csv"
        "flag"
        "fmt"
        "log"
        "math/rand"
        "net/http"
        "os"
        "path/filepath"
        "strconv"
        "sync"
        "time"

        "github.com/aws/aws-sdk-go-v2/aws"
        "github.com/aws/aws-sdk-go-v2/config"
        "github.com/aws/aws-sdk-go-v2/service/s3"
)

// TestConfig holds the configuration for the benchmark
type TestConfig struct {
        Bucket         string
        ExpressBucket  string
        Region         string
        FileSizes      []int  // in KB
        NumIterations  int
        OutputFile     string
        SkipUpload     bool
        SkipDownload   bool
        SkipS3Standard bool
        SkipS3Express  bool
        WarmupCount    int    // Number of warmup requests before measuring
        KeyPrefix      string // Prefix for all test objects
}

// TestResult represents a single test result
type TestResult struct {
        StorageType string
        Operation   string
        FileSize    int
        Duration    time.Duration
        BytesPerSec float64
}

// ObjectKeyCache tracks uploaded objects for reuse in download tests
type ObjectKeyCache struct {
        StandardKeys map[int]string // Cached object keys per size for standard S3
        ExpressKeys  map[int]string // Cached object keys per size for S3 Express
        Mu           sync.Mutex     // Protects the maps
}

func main() {
        config := parseFlags()

        // Setup AWS client with optimized settings
        awsCfg, err := setupAWSClient(config.Region)
        if err != nil {
                log.Fatalf("Failed to set up AWS client: %v", err)
        }

        // Create S3 client - we'll use the same client for both standard and Express
        s3Client := s3.NewFromConfig(awsCfg)

        // Validate that the Express bucket follows the directory bucket naming convention
        if !config.SkipS3Express && !isValidDirectoryBucketName(config.ExpressBucket) {
                log.Printf("Warning: S3 Express bucket name '%s' doesn't appear to follow the directory bucket format.",
                        config.ExpressBucket)
                log.Printf("S3 Express directory bucket names should be in format: 'name--az-id--x-s3' (e.g., 'mybucket--usw2-az1--x-s3')")
        }

        // Create key cache to track uploaded objects
        keyCache := &ObjectKeyCache{
                StandardKeys: make(map[int]string),
                ExpressKeys:  make(map[int]string),
        }

        // Generate random test data for each file size
        testData := make(map[int][]byte)
        for _, size := range config.FileSizes {
                testData[size] = generateRandomData(size * 1024) // Convert KB to bytes
        }

        var results []TestResult

        // Run warmup requests if configured
        if config.WarmupCount > 0 {
                log.Printf("Performing initial warmup with %d requests...", config.WarmupCount)
                runWarmup(s3Client, testData, config, keyCache)

                // Add a short delay after warmup to allow any background operations to complete
                time.Sleep(1 * time.Second)
        }

        // S3 Standard tests
        if !config.SkipS3Standard {
                log.Println("Starting S3 Standard tests...")
                results = append(results, runTests(s3Client, testData, config, "S3Standard", config.Bucket, keyCache)...)
        }

        // S3 Express tests
        if !config.SkipS3Express {
                log.Println("Starting S3 Express One Zone tests...")
                results = append(results, runTests(s3Client, testData, config, "S3Express", config.ExpressBucket, keyCache)...)
        }

        // Write results to CSV
        if err := writeResultsToCSV(results, config.OutputFile); err != nil {
                log.Fatalf("Failed to write results: %v", err)
        }

        log.Printf("Benchmark completed. Results written to %s", config.OutputFile)
}

// Check if a bucket name follows the S3 Express directory bucket naming convention
func isValidDirectoryBucketName(name string) bool {
        // Directory bucket names should end with "--x-s3"
        return len(name) > 6 && name[len(name)-6:] == "--x-s3"
}

func parseFlags() TestConfig {
        bucket := flag.String("bucket", "", "S3 standard bucket name")
        expressBucket := flag.String("express-bucket", "", "S3 Express One Zone directory bucket name (format: name--az-id--x-s3)")
        region := flag.String("region", "us-east-1", "AWS region")
        fileSizesStr := flag.String("file-sizes", "4,16,64,256,1024", "Comma-separated list of file sizes in KB")
        numIterations := flag.Int("iterations", 5, "Number of iterations for each test")
        outputFile := flag.String("output", "s3_benchmark_results.csv", "Output CSV file")
        skipUpload := flag.Bool("skip-upload", false, "Skip upload tests")
        skipDownload := flag.Bool("skip-download", false, "Skip download tests")
        skipS3Standard := flag.Bool("skip-s3-standard", false, "Skip S3 Standard tests")
        skipS3Express := flag.Bool("skip-s3-express", false, "Skip S3 Express One Zone tests")
        warmupCount := flag.Int("warmup", 5, "Number of warmup requests before measuring")
        keyPrefix := flag.String("key-prefix", "benchmark_", "Prefix for S3 object keys")

        flag.Parse()

        if *bucket == "" && !*skipS3Standard {
                log.Fatal("S3 standard bucket name is required")
        }

        if *expressBucket == "" && !*skipS3Express {
                log.Fatal("S3 Express One Zone bucket name is required")
        }

        // Parse file sizes
        var fileSizes []int
        for _, size := range splitCSV(*fileSizesStr) {
                sizeInt, err := strconv.Atoi(size)
                if err != nil {
                        log.Fatalf("Invalid file size: %s", size)
                }
                fileSizes = append(fileSizes, sizeInt)
        }

        return TestConfig{
                Bucket:         *bucket,
                ExpressBucket:  *expressBucket,
                Region:         *region,
                FileSizes:      fileSizes,
                NumIterations:  *numIterations,
                OutputFile:     *outputFile,
                SkipUpload:     *skipUpload,
                SkipDownload:   *skipDownload,
                SkipS3Standard: *skipS3Standard,
                SkipS3Express:  *skipS3Express,
                WarmupCount:    *warmupCount,
                KeyPrefix:      *keyPrefix,
        }
}

func splitCSV(s string) []string {
        var result []string
        for _, v := range bytes.Split([]byte(s), []byte(",")) {
                result = append(result, string(bytes.TrimSpace(v)))
        }
        return result
}

func setupAWSClient(region string) (aws.Config, error) {
        // Create an optimized HTTP client with increased connection pool
        httpClient := &http.Client{
                Transport: &http.Transport{
                        MaxIdleConns:        100,
                        MaxIdleConnsPerHost: 100,
                        IdleConnTimeout:     90 * time.Second,
                },
                Timeout: 30 * time.Second,
        }

        // Load config with optimization settings
        return config.LoadDefaultConfig(context.TODO(),
                config.WithRegion(region),
                config.WithRetryMaxAttempts(1), // Disable retries for benchmarking
                config.WithHTTPClient(httpClient),
        )
}

func generateRandomData(size int) []byte {
        data := make([]byte, size)
        rand.Read(data)
        return data
}

func runWarmup(client *s3.Client, testData map[int][]byte, config TestConfig, keyCache *ObjectKeyCache) {
        // Perform warmup requests to establish connections
        for storageType, bucketName := range map[string]string{
                "S3Standard": config.Bucket,
                "S3Express":  config.ExpressBucket,
        } {
                if (storageType == "S3Standard" && config.SkipS3Standard) ||
                   (storageType == "S3Express" && config.SkipS3Express) {
                        continue
                }

                // Use the smallest file size for warmup
                smallestSize := config.FileSizes[0]
                data := testData[smallestSize]

                for i := 0; i < config.WarmupCount; i++ {
                        // Warmup upload
                        if !config.SkipUpload {
                                key := fmt.Sprintf("%swarmup_%s_%d_kb_%d", config.KeyPrefix, storageType, smallestSize, i)

                                input := &s3.PutObjectInput{
                                        Bucket: aws.String(bucketName),
                                        Key:    aws.String(key),
                                        Body:   bytes.NewReader(data),
                                }

                                _, _ = client.PutObject(context.TODO(), input)

                                // Cache the first key for download warmup
                                if i == 0 {
                                        keyCache.Mu.Lock()
                                        if storageType == "S3Standard" {
                                                keyCache.StandardKeys[smallestSize] = key
                                        } else {
                                                keyCache.ExpressKeys[smallestSize] = key
                                        }
                                        keyCache.Mu.Unlock()
                                }
                        }

                        // Warmup download
                        if !config.SkipDownload {
                                // Use the key we just uploaded
                                key := fmt.Sprintf("%swarmup_%s_%d_kb_%d", config.KeyPrefix, storageType, smallestSize, 0)

                                input := &s3.GetObjectInput{
                                        Bucket: aws.String(bucketName),
                                        Key:    aws.String(key),
                                }

                                resp, _ := client.GetObject(context.TODO(), input)
                                if resp != nil && resp.Body != nil {
                                        resp.Body.Close()
                                }
                        }
                }
        }
}

func runTests(client *s3.Client, testData map[int][]byte, config TestConfig, storageType, bucketName string, keyCache *ObjectKeyCache) []TestResult {
        var results []TestResult

        for _, size := range config.FileSizes {
                log.Printf("Testing %s with file size %d KB", storageType, size)

                // Do a warmup for each file size to ensure connection is established
                log.Printf("Performing warmup for %s with file size %d KB", storageType, size)
                warmupKey := fmt.Sprintf("%swarmup_pre_test_%d_kb_%d", config.KeyPrefix, size, time.Now().UnixNano())

                // Warmup upload
                _, err := client.PutObject(context.TODO(), &s3.PutObjectInput{
                        Bucket: aws.String(bucketName),
                        Key:    aws.String(warmupKey),
                        Body:   bytes.NewReader(testData[size]),
                })

                if err != nil {
                        log.Printf("Warmup upload error: %v", err)
                } else {
                        // Warmup download
                        resp, err := client.GetObject(context.TODO(), &s3.GetObjectInput{
                                Bucket: aws.String(bucketName),
                                Key:    aws.String(warmupKey),
                        })

                        if err != nil {
                                log.Printf("Warmup download error: %v", err)
                        } else if resp != nil && resp.Body != nil {
                                // Drain and close the body
                                buf := new(bytes.Buffer)
                                buf.ReadFrom(resp.Body)
                                resp.Body.Close()
                        }
                }

                // Small delay to ensure the connection is fully established
                time.Sleep(500 * time.Millisecond)

                // Upload tests
                if !config.SkipUpload {
                        for i := 0; i < config.NumIterations; i++ {
                                result := runUploadTest(client, testData[size], size, storageType, bucketName, config.KeyPrefix)

                                // Cache the key for future downloads if not an error
                                if result.Operation != "Upload-Error" {
                                        keyCache.Mu.Lock()
                                        if storageType == "S3Standard" {
                                                keyCache.StandardKeys[size] = result.Operation[7:] // Strip "Upload-" prefix
                                        } else {
                                                keyCache.ExpressKeys[size] = result.Operation[7:] // Strip "Upload-" prefix
                                        }
                                        keyCache.Mu.Unlock()

                                        // Reset the operation name for results
                                        result.Operation = "Upload"
                                }

                                results = append(results, result)
                        }
                }

                // Small delay between upload and download tests
                time.Sleep(500 * time.Millisecond)

                // Download tests
                if !config.SkipDownload {
                        // Get the cached key for this size and storage type
                        var key string
                        keyCache.Mu.Lock()
                        if storageType == "S3Standard" {
                                key = keyCache.StandardKeys[size]
                        } else {
                                key = keyCache.ExpressKeys[size]
                        }
                        keyCache.Mu.Unlock()

                        // If we don't have a cached key, ensure the file exists
                        if key == "" && !config.SkipUpload {
                                // Create a key and upload the object
                                key = fmt.Sprintf("%s%d_kb_%s.bin", config.KeyPrefix, size, time.Now().Format("20060102150405"))

                                // Upload the file
                                input := &s3.PutObjectInput{
                                        Bucket: aws.String(bucketName),
                                        Key:    aws.String(key),
                                        Body:   bytes.NewReader(testData[size]),
                                }

                                _, err := client.PutObject(context.TODO(), input)

                                if err != nil {
                                        log.Printf("Error uploading file for download tests: %v", err)
                                        continue
                                }

                                // Cache the key
                                keyCache.Mu.Lock()
                                if storageType == "S3Standard" {
                                        keyCache.StandardKeys[size] = key
                                } else {
                                        keyCache.ExpressKeys[size] = key
                                }
                                keyCache.Mu.Unlock()
                        }

                        // Run sequential download tests
                        for i := 0; i < config.NumIterations; i++ {
                                results = append(results, runDownloadTest(client, size, storageType, bucketName, key))
                        }
                }
        }

        return results
}

func runUploadTest(client *s3.Client, data []byte, sizeInKB int, storageType, bucketName, keyPrefix string) TestResult {
        // Create a unique key using the current timestamp
        key := fmt.Sprintf("%s%d_kb_%s.bin", keyPrefix, sizeInKB, time.Now().Format("20060102150405"))

        input := &s3.PutObjectInput{
                Bucket: aws.String(bucketName),
                Key:    aws.String(key),
                Body:   bytes.NewReader(data),
        }

        start := time.Now()
        _, err := client.PutObject(context.TODO(), input)
        duration := time.Since(start)

        if err != nil {
                log.Printf("Upload error for %s (%d KB): %v", storageType, sizeInKB, err)
                return TestResult{
                        StorageType: storageType,
                        Operation:   "Upload-Error",
                        FileSize:    sizeInKB,
                        Duration:    duration,
                        BytesPerSec: 0,
                }
        }

        bytesPerSec := float64(len(data)) / duration.Seconds()

        // Return the key in the operation field temporarily
        // This will be used to cache the key for download tests
        return TestResult{
                StorageType: storageType,
                Operation:   "Upload-" + key,
                FileSize:    sizeInKB,
                Duration:    duration,
                BytesPerSec: bytesPerSec,
        }
}

func runDownloadTest(client *s3.Client, sizeInKB int, storageType, bucketName, key string) TestResult {
        if key == "" {
                return TestResult{
                        StorageType: storageType,
                        Operation:   "Download-Error-NoKey",
                        FileSize:    sizeInKB,
                        Duration:    0,
                        BytesPerSec: 0,
                }
        }

        input := &s3.GetObjectInput{
                Bucket: aws.String(bucketName),
                Key:    aws.String(key),
        }

        start := time.Now()
        resp, err := client.GetObject(context.TODO(), input)

        if err != nil {
                duration := time.Since(start)
                log.Printf("Download error for %s (%d KB): %v", storageType, sizeInKB, err)
                return TestResult{
                        StorageType: storageType,
                        Operation:   "Download-Error",
                        FileSize:    sizeInKB,
                        Duration:    duration,
                        BytesPerSec: 0,
                }
        }

        // Read the body to ensure we actually download the data
        buf := new(bytes.Buffer)
        bytesDownloaded, err := buf.ReadFrom(resp.Body)
        resp.Body.Close()

        duration := time.Since(start)

        if err != nil {
                log.Printf("Download read error for %s (%d KB): %v", storageType, sizeInKB, err)
                return TestResult{
                        StorageType: storageType,
                        Operation:   "Download-Error",
                        FileSize:    sizeInKB,
                        Duration:    duration,
                        BytesPerSec: 0,
                }
        }

        bytesPerSec := float64(bytesDownloaded) / duration.Seconds()

        return TestResult{
                StorageType: storageType,
                Operation:   "Download",
                FileSize:    sizeInKB,
                Duration:    duration,
                BytesPerSec: bytesPerSec,
        }
}

func writeResultsToCSV(results []TestResult, filename string) error {
        // Create directory if it doesn't exist
        dir := filepath.Dir(filename)
        if dir != "." && dir != "" {
                if err := os.MkdirAll(dir, 0755); err != nil {
                        return err
                }
        }

        file, err := os.Create(filename)
        if err != nil {
                return err
        }
        defer file.Close()

        writer := csv.NewWriter(file)
        defer writer.Flush()

        // Write header
        header := []string{"StorageType", "Operation", "FileSize_KB", "Duration_ms", "Throughput_MBps"}
        if err := writer.Write(header); err != nil {
                return err
        }

        // Write results
        for _, result := range results {
                durationMs := result.Duration.Milliseconds()
                throughputMBps := result.BytesPerSec / (1024 * 1024) // Convert bytes/sec to MB/sec

                row := []string{
                        result.StorageType,
                        result.Operation,
                        strconv.Itoa(result.FileSize),
                        strconv.FormatInt(durationMs, 10),
                        strconv.FormatFloat(throughputMBps, 'f', 2, 64),
                }

                if err := writer.Write(row); err != nil {
                        return err
                }
        }

        return nil
}