//date: 2025-10-29T17:01:06Z
//url: https://api.github.com/gists/f326d450b1efb9d736fa6ff446ae0065
//owner: https://api.github.com/users/arturAtDeliveroo

// File: pkg/redisconn/generic_swr_cache.go
// Surgical change: Replace channel queue with deduplicated map + signal

// In GenericSWRCache struct (around line 36), REPLACE:
//   refreshQueue chan string
// WITH:
refreshQueue   map[string]struct{} // Deduplicated set of keys needing refresh
refreshQueueMu sync.Mutex
refreshSignal  chan struct{} // Signal workers that queue has items

// In NewGenericSWRCache (around line 149), REPLACE:
//   refreshQueue: make(chan string, cfg.RefreshQueueSize),
// WITH:
refreshQueue:  make(map[string]struct{}),
refreshSignal: make(chan struct{}, 1),

// In GetBulk (around lines 240-250), REPLACE entire queueing section:
//   if !refreshQueued[key] {
//       select {
//       case c.refreshQueue <- key:
//           refreshQueued[key] = true
//       default:
//       }
//   }
// WITH:
c.queueRefresh(key)

// ADD new method after GetBulkWithFetch:
func (c *GenericSWRCache) queueRefresh(key string) {
	c.refreshQueueMu.Lock()
	defer c.refreshQueueMu.Unlock()

	if len(c.refreshQueue) >= c.config.RefreshQueueSize {
		if c.statsd != nil {
			c.statsd.Count("generic_cache.refresh_queue_full", 1, 1)
		}
		return
	}

	if _, exists := c.refreshQueue[key]; !exists {
		c.refreshQueue[key] = struct{}{}
		
		select {
		case c.refreshSignal <- struct{}{}:
		default:
		}
	}
}

// In refreshWorker (around lines 560-590), REPLACE entire select block:
//   case key := <-c.refreshQueue:
//       batch = append(batch, key)
// WITH:
case <-c.refreshSignal:
	c.refreshQueueMu.Lock()
	for key := range c.refreshQueue {
		batch = append(batch, key)
		delete(c.refreshQueue, key)
		
		if len(batch) >= c.config.RefreshBatchSize {
			break
		}
	}
	c.refreshQueueMu.Unlock()

// In scanAndQueueStaleEntries (around line 718), REPLACE:
//   case c.refreshQueue <- entry.key:
// WITH:
c.queueRefresh(entry.key)
