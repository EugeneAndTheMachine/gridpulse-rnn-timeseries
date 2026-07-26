"""Group consecutive anomaly flags into events."""
import numpy as np
from dataclasses import dataclass


@dataclass
class AnomalyEvent:
    start_idx: int
    end_idx: int
    duration: int
    peak_score: float
    mean_score: float

    def __repr__(self):
        return (
            f"Event(start={self.start_idx}, end={self.end_idx}, "
            f"dur={self.duration}, peak={self.peak_score:.3f})"
        )


def _walk_run(is_anomaly: np.ndarray, start: int, max_gap: int) -> int:
    """Return the end index of the run starting at `start`.

    Two True positions belong to the same run if they are separated by
    ≤ `max_gap` consecutive False positions.
    """
    n = len(is_anomaly)
    end = start
    j = start + 1
    while j < n:
        if is_anomaly[j]:
            end = j
            j += 1
            continue
        # j is False — look ahead up to max_gap positions for the next True.
        probe = j
        while probe < n and probe - j < max_gap and not is_anomaly[probe]:
            probe += 1
        if probe < n and is_anomaly[probe] and probe - j <= max_gap:
            end = probe
            j = probe + 1
        else:
            break
    return end


def group_anomalies(
    is_anomaly: np.ndarray,
    scores: np.ndarray,
    max_gap: int = 2,
    min_duration: int = 1,
) -> list[AnomalyEvent]:
    """
    Group consecutive anomaly points into events.

    Args:
        is_anomaly: (N,) boolean array
        scores: (N,) anomaly scores
        max_gap: cho phép gap tối đa (số điểm False liên tiếp) giữa 2
            anomalies để vẫn thuộc cùng 1 event.
        min_duration: event ngắn hơn số này sẽ bị bỏ (noise filter)
    """
    events: list[AnomalyEvent] = []
    n = len(is_anomaly)
    i = 0

    while i < n:
        if not is_anomaly[i]:
            i += 1
            continue

        start = i
        end = _walk_run(is_anomaly, start, max_gap)

        duration = end - start + 1
        if duration >= min_duration:
            event_scores = scores[start:end + 1]
            events.append(AnomalyEvent(
                start_idx=start,
                end_idx=end,
                duration=duration,
                peak_score=float(event_scores.max()),
                mean_score=float(event_scores.mean()),
            ))
        i = end + 1

    return events


def filter_short_runs(
    is_anomaly: np.ndarray, min_duration: int = 2, max_gap: int = 0
) -> np.ndarray:
    """
    Zero out anomaly runs shorter than `min_duration`.

    Runs are contiguous blocks of True separated by gaps ≤ `max_gap`.
    Returns a boolean array the same shape as input; runs surviving the
    duration cut remain True, everything else is False.
    """
    n = len(is_anomaly)
    out = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if not is_anomaly[i]:
            i += 1
            continue

        start = i
        end = _walk_run(is_anomaly, start, max_gap)

        if (end - start + 1) >= min_duration:
            out[start:end + 1] = True
        i = end + 1

    return out
