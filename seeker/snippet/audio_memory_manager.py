#date: 2025-12-30T17:00:18Z
#url: https://api.github.com/gists/0d5e9d8242d3bb7197dbbf1000cb8931
#owner: https://api.github.com/users/bogged-broker

"""
audio_memory_manager.py

Production-grade memory management system for audio patterns in RL environments.
Handles pattern lifecycle, decay, prioritization, and diversity enforcement.
"""

import time
import json
import sqlite3
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from pathlib import Path


@dataclass
class AudioPattern:
    """Represents a learned audio pattern with metadata."""
    pattern_id: str
    pattern_type: str  # 'tts', 'voice_sync', 'beat', 'niche'
    features: Dict
    performance_score: float
    success_count: int
    failure_count: int
    created_at: float
    last_used: float
    decay_factor: float
    niche: str
    platform: str
    effective_score: float  # Score after decay


class AudioMemoryManager:
    """
    Manages full lifecycle of audio patterns with decay, prioritization, and diversity.
    """
    
    def __init__(
        self,
        db_path: str = "audio_patterns.db",
        decay_rate: float = 0.95,
        decay_interval: int = 3600,  # 1 hour
        min_score_threshold: float = 0.3,
        diversity_weight: float = 0.2,
        recency_weight: float = 0.4,
        performance_weight: float = 0.4
    ):
        """
        Initialize the memory manager.
        
        Args:
            db_path: SQLite database path for persistent storage
            decay_rate: Exponential decay rate per interval (0-1)
            decay_interval: Time in seconds between decay applications
            min_score_threshold: Minimum score to keep pattern active
            diversity_weight: Weight for diversity scoring
            recency_weight: Weight for recency in scoring
            performance_weight: Weight for performance metrics
        """
        self.db_path = db_path
        self.decay_rate = decay_rate
        self.decay_interval = decay_interval
        self.min_score_threshold = min_score_threshold
        self.diversity_weight = diversity_weight
        self.recency_weight = recency_weight
        self.performance_weight = performance_weight
        
        # In-memory cache for fast access
        self.pattern_cache: Dict[str, AudioPattern] = {}
        self.niche_counts: Dict[str, int] = defaultdict(int)
        self.platform_counts: Dict[str, int] = defaultdict(int)
        self.type_counts: Dict[str, int] = defaultdict(int)
        
        self._init_database()
        self._load_patterns()
        self.last_decay_time = time.time()
    
    def _init_database(self):
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                features TEXT NOT NULL,
                performance_score REAL NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_used REAL NOT NULL,
                decay_factor REAL DEFAULT 1.0,
                niche TEXT NOT NULL,
                platform TEXT NOT NULL,
                effective_score REAL NOT NULL,
                active INTEGER DEFAULT 1
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_effective_score 
            ON patterns(effective_score DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_niche_platform 
            ON patterns(niche, platform)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pattern_type 
            ON patterns(pattern_type)
        """)
        
        conn.commit()
        conn.close()
    
    def _load_patterns(self):
        """Load active patterns from database into memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pattern_id, pattern_type, features, performance_score,
                   success_count, failure_count, created_at, last_used,
                   decay_factor, niche, platform, effective_score
            FROM patterns WHERE active = 1
        """)
        
        for row in cursor.fetchall():
            pattern = AudioPattern(
                pattern_id=row[0],
                pattern_type=row[1],
                features=json.loads(row[2]),
                performance_score=row[3],
                success_count=row[4],
                failure_count=row[5],
                created_at=row[6],
                last_used=row[7],
                decay_factor=row[8],
                niche=row[9],
                platform=row[10],
                effective_score=row[11]
            )
            self.pattern_cache[pattern.pattern_id] = pattern
            self.niche_counts[pattern.niche] += 1
            self.platform_counts[pattern.platform] += 1
            self.type_counts[pattern.pattern_type] += 1
        
        conn.close()
        print(f"Loaded {len(self.pattern_cache)} active patterns from database")
    
    def record_pattern_success(
        self,
        pattern_id: str,
        performance_score: float,
        pattern_type: str = "tts",
        features: Optional[Dict] = None,
        niche: str = "general",
        platform: str = "default"
    ) -> bool:
        """
        Record a successful pattern usage or create new pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            performance_score: Score from 0-1 indicating performance
            pattern_type: Type of pattern (tts, voice_sync, beat, niche)
            features: Feature dictionary for new patterns
            niche: Content niche/category
            platform: Platform identifier
            
        Returns:
            True if pattern was recorded successfully
        """
        current_time = time.time()
        
        if pattern_id in self.pattern_cache:
            # Update existing pattern
            pattern = self.pattern_cache[pattern_id]
            pattern.success_count += 1
            pattern.last_used = current_time
            
            # Exponential moving average for performance score
            alpha = 0.3  # Learning rate for score updates
            pattern.performance_score = (
                alpha * performance_score + 
                (1 - alpha) * pattern.performance_score
            )
            
            # Boost decay factor for successful recent usage
            pattern.decay_factor = min(1.0, pattern.decay_factor * 1.05)
            
        else:
            # Create new pattern
            if features is None:
                features = {}
            
            pattern = AudioPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                features=features,
                performance_score=performance_score,
                success_count=1,
                failure_count=0,
                created_at=current_time,
                last_used=current_time,
                decay_factor=1.0,
                niche=niche,
                platform=platform,
                effective_score=performance_score
            )
            
            self.pattern_cache[pattern_id] = pattern
            self.niche_counts[niche] += 1
            self.platform_counts[platform] += 1
            self.type_counts[pattern_type] += 1
        
        # Recalculate effective score
        self._update_effective_score(pattern)
        
        # Persist to database
        self._save_pattern(pattern)
        
        return True
    
    def record_pattern_failure(self, pattern_id: str):
        """Record a pattern failure to adjust its scoring."""
        if pattern_id not in self.pattern_cache:
            return
        
        pattern = self.pattern_cache[pattern_id]
        pattern.failure_count += 1
        
        # Penalize performance score
        penalty = 0.1 * (pattern.failure_count / (pattern.success_count + 1))
        pattern.performance_score = max(0, pattern.performance_score - penalty)
        
        self._update_effective_score(pattern)
        self._save_pattern(pattern)
    
    def _update_effective_score(self, pattern: AudioPattern):
        """Calculate effective score with decay, recency, and diversity."""
        current_time = time.time()
        
        # Time-based decay
        time_since_use = current_time - pattern.last_used
        time_decay = np.exp(-time_since_use / (7 * 24 * 3600))  # 7-day half-life
        
        # Recency component
        recency_score = time_decay
        
        # Performance component
        success_rate = pattern.success_count / max(1, pattern.success_count + pattern.failure_count)
        performance_score = pattern.performance_score * success_rate
        
        # Diversity penalty (reduce score if niche/platform overrepresented)
        total_patterns = len(self.pattern_cache)
        niche_ratio = self.niche_counts[pattern.niche] / max(1, total_patterns)
        platform_ratio = self.platform_counts[pattern.platform] / max(1, total_patterns)
        diversity_score = 1.0 - (niche_ratio + platform_ratio) / 2
        
        # Weighted combination
        pattern.effective_score = (
            self.recency_weight * recency_score +
            self.performance_weight * performance_score +
            self.diversity_weight * diversity_score
        ) * pattern.decay_factor
    
    def get_active_patterns(
        self,
        pattern_type: Optional[str] = None,
        niche: Optional[str] = None,
        platform: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[AudioPattern]:
        """
        Retrieve active patterns with optional filtering and ranking.
        
        Args:
            pattern_type: Filter by pattern type
            niche: Filter by niche
            platform: Filter by platform
            top_k: Return only top K patterns by effective score
            min_score: Minimum effective score threshold
            
        Returns:
            List of AudioPattern objects sorted by effective score
        """
        # Auto-decay if needed
        if time.time() - self.last_decay_time > self.decay_interval:
            self.decay_old_patterns()
        
        patterns = list(self.pattern_cache.values())
        
        # Apply filters
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        if niche:
            patterns = [p for p in patterns if p.niche == niche]
        if platform:
            patterns = [p for p in patterns if p.platform == platform]
        
        # Score threshold
        threshold = min_score if min_score is not None else self.min_score_threshold
        patterns = [p for p in patterns if p.effective_score >= threshold]
        
        # Sort by effective score
        patterns.sort(key=lambda p: p.effective_score, reverse=True)
        
        # Apply top-k limit
        if top_k:
            patterns = patterns[:top_k]
        
        return patterns
    
    def decay_old_patterns(self) -> Dict[str, int]:
        """
        Apply exponential decay to all patterns and remove stale ones.
        
        Returns:
            Dictionary with decay statistics
        """
        current_time = time.time()
        time_since_last_decay = current_time - self.last_decay_time
        decay_periods = time_since_last_decay / self.decay_interval
        
        deprecated_patterns = []
        stats = {
            'total_processed': 0,
            'deprecated': 0,
            'active_remaining': 0
        }
        
        for pattern_id, pattern in list(self.pattern_cache.items()):
            stats['total_processed'] += 1
            
            # Apply exponential decay
            pattern.decay_factor *= (self.decay_rate ** decay_periods)
            
            # Recalculate effective score
            self._update_effective_score(pattern)
            
            # Check if pattern should be deprecated
            if pattern.effective_score < self.min_score_threshold:
                deprecated_patterns.append(pattern_id)
                stats['deprecated'] += 1
            else:
                stats['active_remaining'] += 1
                self._save_pattern(pattern)
        
        # Remove deprecated patterns
        for pattern_id in deprecated_patterns:
            self._deprecate_pattern(pattern_id)
        
        self.last_decay_time = current_time
        
        print(f"Decay complete: {stats['deprecated']} patterns deprecated, "
              f"{stats['active_remaining']} remain active")
        
        return stats
    
    def _deprecate_pattern(self, pattern_id: str):
        """Mark pattern as inactive and remove from cache."""
        if pattern_id not in self.pattern_cache:
            return
        
        pattern = self.pattern_cache[pattern_id]
        
        # Update counts
        self.niche_counts[pattern.niche] -= 1
        self.platform_counts[pattern.platform] -= 1
        self.type_counts[pattern.pattern_type] -= 1
        
        # Remove from cache
        del self.pattern_cache[pattern_id]
        
        # Mark inactive in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE patterns SET active = 0 WHERE pattern_id = ?", (pattern_id,))
        conn.commit()
        conn.close()
    
    def _save_pattern(self, pattern: AudioPattern):
        """Persist pattern to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO patterns 
            (pattern_id, pattern_type, features, performance_score, success_count,
             failure_count, created_at, last_used, decay_factor, niche, platform,
             effective_score, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            pattern.pattern_id,
            pattern.pattern_type,
            json.dumps(pattern.features),
            pattern.performance_score,
            pattern.success_count,
            pattern.failure_count,
            pattern.created_at,
            pattern.last_used,
            pattern.decay_factor,
            pattern.niche,
            pattern.platform,
            pattern.effective_score
        ))
        
        conn.commit()
        conn.close()
    
    def get_diversity_report(self) -> Dict:
        """Generate report on pattern diversity across dimensions."""
        return {
            'total_patterns': len(self.pattern_cache),
            'by_niche': dict(self.niche_counts),
            'by_platform': dict(self.platform_counts),
            'by_type': dict(self.type_counts),
            'avg_effective_score': np.mean([p.effective_score for p in self.pattern_cache.values()]),
            'score_distribution': self._get_score_distribution()
        }
    
    def _get_score_distribution(self) -> Dict[str, int]:
        """Get distribution of patterns by score ranges."""
        scores = [p.effective_score for p in self.pattern_cache.values()]
        return {
            '0.0-0.2': sum(1 for s in scores if 0.0 <= s < 0.2),
            '0.2-0.4': sum(1 for s in scores if 0.2 <= s < 0.4),
            '0.4-0.6': sum(1 for s in scores if 0.4 <= s < 0.6),
            '0.6-0.8': sum(1 for s in scores if 0.6 <= s < 0.8),
            '0.8-1.0': sum(1 for s in scores if 0.8 <= s <= 1.0)
        }
    
    def force_diversity_rebalance(self, target_diversity: float = 0.3):
        """
        Enforce diversity by deprecating overrepresented patterns.
        
        Args:
            target_diversity: Target maximum ratio for any single category
        """
        total = len(self.pattern_cache)
        max_per_category = int(total * target_diversity)
        
        # Rebalance niches
        for niche, count in list(self.niche_counts.items()):
            if count > max_per_category:
                # Find lowest scoring patterns in this niche
                niche_patterns = [
                    (pid, p) for pid, p in self.pattern_cache.items()
                    if p.niche == niche
                ]
                niche_patterns.sort(key=lambda x: x[1].effective_score)
                
                # Deprecate excess
                for i in range(count - max_per_category):
                    self._deprecate_pattern(niche_patterns[i][0])
        
        # Rebalance platforms
        for platform, count in list(self.platform_counts.items()):
            if count > max_per_category:
                platform_patterns = [
                    (pid, p) for pid, p in self.pattern_cache.items()
                    if p.platform == platform
                ]
                platform_patterns.sort(key=lambda x: x[1].effective_score)
                
                for i in range(count - max_per_category):
                    self._deprecate_pattern(platform_patterns[i][0])
    
    def export_top_patterns(self, output_path: str, top_k: int = 100):
        """Export top patterns to JSON for model integration."""
        patterns = self.get_active_patterns(top_k=top_k)
        
        export_data = {
            'timestamp': time.time(),
            'count': len(patterns),
            'patterns': [asdict(p) for p in patterns]
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(patterns)} patterns to {output_path}")


# Example RL Integration
class RLAudioIntegration:
    """Example integration with RL training loop."""
    
    def __init__(self, memory_manager: AudioMemoryManager):
        self.memory = memory_manager
    
    def update_from_episode(self, episode_data: Dict):
        """
        Update memory based on RL episode results.
        
        Args:
            episode_data: Dictionary containing episode results
                - pattern_id: Pattern identifier
                - reward: Episode reward signal
                - pattern_type: Type of pattern used
                - features: Pattern features
                - metadata: Additional context (niche, platform, etc.)
        """
        pattern_id = episode_data['pattern_id']
        reward = episode_data['reward']
        
        # Normalize reward to 0-1 score
        performance_score = max(0, min(1, (reward + 1) / 2))
        
        # Record success or failure
        if performance_score > 0.5:
            self.memory.record_pattern_success(
                pattern_id=pattern_id,
                performance_score=performance_score,
                pattern_type=episode_data.get('pattern_type', 'tts'),
                features=episode_data.get('features', {}),
                niche=episode_data.get('metadata', {}).get('niche', 'general'),
                platform=episode_data.get('metadata', {}).get('platform', 'default')
            )
        else:
            self.memory.record_pattern_failure(pattern_id)
    
    def get_policy_patterns(self, context: Dict) -> List[AudioPattern]:
        """Retrieve patterns for policy decision-making."""
        return self.memory.get_active_patterns(
            pattern_type=context.get('type'),
            niche=context.get('niche'),
            platform=context.get('platform'),
            top_k=20
        )


if __name__ == "__main__":
    # Example usage
    manager = AudioMemoryManager()
    
    # Record some patterns
    manager.record_pattern_success(
        pattern_id="tts_energetic_001",
        performance_score=0.85,
        pattern_type="tts",
        features={"tempo": "fast", "energy": "high"},
        niche="fitness",
        platform="tiktok"
    )
    
    manager.record_pattern_success(
        pattern_id="voice_sync_smooth_001",
        performance_score=0.92,
        pattern_type="voice_sync",
        features={"smoothness": 0.9, "latency": 50},
        niche="asmr",
        platform="youtube"
    )
    
    # Get active patterns
    top_patterns = manager.get_active_patterns(top_k=10)
    print(f"\nTop {len(top_patterns)} active patterns:")
    for p in top_patterns:
        print(f"  {p.pattern_id}: {p.effective_score:.3f} (type={p.pattern_type}, niche={p.niche})")
    
    # Diversity report
    print("\nDiversity Report:")
    report = manager.get_diversity_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # Apply decay
    print("\nApplying decay...")
    stats = manager.decay_old_patterns()
    print(f"  Deprecated: {stats['deprecated']}")
    print(f"  Active: {stats['active_remaining']}")