//date: 2025-10-29T17:01:06Z
//url: https://api.github.com/gists/f326d450b1efb9d736fa6ff446ae0065
//owner: https://api.github.com/users/arturAtDeliveroo

// File: pkg/redisconn/generic_swr_cache.go
// Surgical change: Add memory breakdown metrics by cache state

// In reportDataFreshness method (around line 808), REPLACE entire function:
func (c *GenericSWRCache) reportDataFreshness() {
	now := time.Now()
	staleThreshold := time.Duration(c.config.StaleAfterSeconds) * time.Second
	invalidThreshold := time.Duration(c.config.InvalidAfterSeconds) * time.Second
	zombieThreshold := 5 * time.Minute

	var (
		totalEntries int
		freshEntries int
		staleEntries int
		invalidEntries int
		zombieEntries int
		sumAge time.Duration
		
		// Memory breakdown
		freshMemory int64
		staleMemory int64
		invalidMemory int64
		zombieMemory int64
	)

	c.mu.RLock()
	for _, entry := range c.entries {
		age := now.Sub(entry.LastRefreshed)
		totalEntries++
		sumAge += age
		
		timeSinceAccess := now.Sub(entry.GetLastAccessed())

		if age < staleThreshold {
			// Fresh
			freshEntries++
			freshMemory += entry.sizeBytes
		} else if age < invalidThreshold {
			// Stale
			staleEntries++
			staleMemory += entry.sizeBytes
		} else {
			// Invalid
			invalidEntries++
			invalidMemory += entry.sizeBytes
			
			if timeSinceAccess > zombieThreshold {
				zombieEntries++
				zombieMemory += entry.sizeBytes
			}
		}
	}
	currentMemory := c.currentMemoryBytes
	c.mu.RUnlock()

	if totalEntries == 0 {
		return
	}

	avgAge := sumAge / time.Duration(totalEntries)
	unusedMemory := c.maxMemoryBytes - currentMemory

	// Report average age
	c.statsd.Distribution("generic_cache.bg_tracker.data_age_seconds", avgAge.Seconds(), 1)

	// Report entry counts by state
	c.statsd.Distribution("generic_cache.bg_tracker.entries_by_state", float64(freshEntries), 1, "state", "fresh")
	c.statsd.Distribution("generic_cache.bg_tracker.entries_by_state", float64(staleEntries), 1, "state", "stale")
	c.statsd.Distribution("generic_cache.bg_tracker.entries_by_state", float64(invalidEntries), 1, "state", "invalid")
	c.statsd.Distribution("generic_cache.bg_tracker.entries_by_state", float64(zombieEntries), 1, "state", "zombie")

	// Report memory usage by state (MB)
	c.statsd.Distribution("generic_cache.bg_tracker.memory_by_state_mb", float64(freshMemory)/(1024*1024), 1, "state", "fresh")
	c.statsd.Distribution("generic_cache.bg_tracker.memory_by_state_mb", float64(staleMemory)/(1024*1024), 1, "state", "stale")
	c.statsd.Distribution("generic_cache.bg_tracker.memory_by_state_mb", float64(invalidMemory)/(1024*1024), 1, "state", "invalid")
	c.statsd.Distribution("generic_cache.bg_tracker.memory_by_state_mb", float64(zombieMemory)/(1024*1024), 1, "state", "zombie")
	c.statsd.Distribution("generic_cache.bg_tracker.memory_by_state_mb", float64(unusedMemory)/(1024*1024), 1, "state", "unused")

	// Report percentages
	freshPct := float64(freshEntries) / float64(totalEntries) * 100
	stalePct := float64(staleEntries) / float64(totalEntries) * 100
	invalidPct := float64(invalidEntries) / float64(totalEntries) * 100
	zombiePct := float64(zombieEntries) / float64(totalEntries) * 100
	
	c.statsd.Distribution("generic_cache.bg_tracker.state_percentage", freshPct, 1, "state", "fresh")
	c.statsd.Distribution("generic_cache.bg_tracker.state_percentage", stalePct, 1, "state", "stale")
	c.statsd.Distribution("generic_cache.bg_tracker.state_percentage", invalidPct, 1, "state", "invalid")
	c.statsd.Distribution("generic_cache.bg_tracker.state_percentage", zombiePct, 1, "state", "zombie")
	
	memoryUtilizationPct := float64(currentMemory) / float64(c.maxMemoryBytes) * 100
	c.statsd.Distribution("generic_cache.bg_tracker.memory_utilization_pct", memoryUtilizationPct, 1)
}
