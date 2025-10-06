"""
Time utilities for temporal feature extraction and bucketing.

This module provides utilities for processing temporal information including
time-of-day bucketing, delta-time calculations, and temporal feature generation.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import math


def create_time_buckets(bucket_minutes: int = 30) -> List[str]:
    """Create time-of-day bucket labels."""
    buckets = []
    for hour in range(24):
        for minute in range(0, 60, bucket_minutes):
            time_str = f"{hour:02d}:{minute:02d}"
            buckets.append(time_str)
    return buckets


def get_tod_bucket(timestamp: datetime, bucket_minutes: int = 30) -> str:
    """Get time-of-day bucket for a timestamp."""
    # Round down to nearest bucket
    bucket_minute = (timestamp.minute // bucket_minutes) * bucket_minutes
    return f"{timestamp.hour:02d}:{bucket_minute:02d}"


def get_tod_bucket_name(timestamp: datetime) -> str:
    """Get descriptive time-of-day bucket name."""
    hour = timestamp.hour

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    elif 21 <= hour < 24:
        return "night"
    else:  # 0 <= hour < 5
        return "late_night"


def get_delta_t_bucket(delta_seconds: float, log_scale: bool = True) -> str:
    """Get delta-time bucket for time gaps between events."""
    if delta_seconds <= 0:
        return "0s"

    if log_scale:
        # Log-scale buckets
        log_seconds = math.log10(delta_seconds)

        if log_seconds < 0:  # < 1 second
            return "<1s"
        elif log_seconds < 1:  # 1-10 seconds
            return "1-10s"
        elif log_seconds < 2:  # 10-100 seconds (1-2 minutes)
            return "10s-2m"
        elif log_seconds < 3:  # 100-1000 seconds (2-17 minutes)
            return "2-17m"
        elif log_seconds < 4:  # 1000-10000 seconds (17 minutes - 3 hours)
            return "17m-3h"
        else:  # > 3 hours
            return ">3h"
    else:
        # Linear buckets
        minutes = delta_seconds / 60

        if minutes < 1:
            return "<1m"
        elif minutes < 5:
            return "1-5m"
        elif minutes < 15:
            return "5-15m"
        elif minutes < 60:
            return "15-60m"
        else:
            return ">1h"


def extract_temporal_features(timestamp: datetime) -> Dict[str, any]:
    """Extract various temporal features from a timestamp."""
    return {
        'hour': timestamp.hour,
        'minute': timestamp.minute,
        'day_of_week': timestamp.weekday(),  # 0=Monday, 6=Sunday
        'day_of_week_name': timestamp.strftime('%A'),
        'month': timestamp.month,
        'day_of_month': timestamp.day,
        'is_weekend': timestamp.weekday() >= 5,
        'tod_bucket': get_tod_bucket(timestamp),
        'tod_name': get_tod_bucket_name(timestamp),

        # Cyclical encodings
        'hour_sin': math.sin(2 * math.pi * timestamp.hour / 24),
        'hour_cos': math.cos(2 * math.pi * timestamp.hour / 24),
        'dow_sin': math.sin(2 * math.pi * timestamp.weekday() / 7),
        'dow_cos': math.cos(2 * math.pi * timestamp.weekday() / 7),
        'month_sin': math.sin(2 * math.pi * timestamp.month / 12),
        'month_cos': math.cos(2 * math.pi * timestamp.month / 12)
    }


def compute_time_gaps(timestamps: List[datetime]) -> List[float]:
    """Compute time gaps in seconds between consecutive timestamps."""
    if len(timestamps) < 2:
        return []

    gaps = []
    for i in range(1, len(timestamps)):
        gap = (timestamps[i] - timestamps[i-1]).total_seconds()
        gaps.append(gap)

    return gaps


def bucket_time_gaps(time_gaps: List[float], log_scale: bool = True) -> List[str]:
    """Convert time gaps to bucket labels."""
    return [get_delta_t_bucket(gap, log_scale) for gap in time_gaps]


def get_time_context(timestamp: datetime) -> Dict[str, str]:
    """Get contextual time description for caption generation."""
    hour = timestamp.hour
    minute = timestamp.minute

    # Time of day context
    if hour < 6:
        time_context = "late night" if hour < 3 else "early morning"
    elif hour < 12:
        time_context = "morning"
    elif hour < 17:
        time_context = "afternoon"
    elif hour < 21:
        time_context = "evening"
    else:
        time_context = "night"

    # More specific descriptions
    if 23 <= hour or hour < 2:
        specific = "middle of the night"
    elif 2 <= hour < 6:
        specific = "very early morning"
    elif 6 <= hour < 9:
        specific = "early morning"
    elif 9 <= hour < 12:
        specific = "late morning"
    elif 12 <= hour < 14:
        specific = "midday"
    elif 14 <= hour < 17:
        specific = "mid afternoon"
    elif 17 <= hour < 20:
        specific = "early evening"
    elif 20 <= hour < 23:
        specific = "late evening"
    else:
        specific = time_context

    return {
        'general': time_context,
        'specific': specific,
        'day_of_week': timestamp.strftime('%A')
    }


def create_temporal_embedding(timestamp: datetime, embedding_dim: int = 16) -> np.ndarray:
    """Create a temporal embedding vector for a timestamp."""
    features = []

    # Hour embedding (cyclical)
    hour_angle = 2 * np.pi * timestamp.hour / 24
    features.extend([np.sin(hour_angle), np.cos(hour_angle)])

    # Day of week embedding (cyclical)
    dow_angle = 2 * np.pi * timestamp.weekday() / 7
    features.extend([np.sin(dow_angle), np.cos(dow_angle)])

    # Month embedding (cyclical)
    month_angle = 2 * np.pi * timestamp.month / 12
    features.extend([np.sin(month_angle), np.cos(month_angle)])

    # Day of month embedding (cyclical)
    dom_angle = 2 * np.pi * timestamp.day / 31
    features.extend([np.sin(dom_angle), np.cos(dom_angle)])

    # Pad or truncate to desired dimension
    features = np.array(features)
    if len(features) < embedding_dim:
        # Pad with zeros
        padding = np.zeros(embedding_dim - len(features))
        features = np.concatenate([features, padding])
    else:
        # Truncate
        features = features[:embedding_dim]

    return features
