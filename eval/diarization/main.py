"""Main Diarization evaluation implementation.

This module implements the core evaluation metrics for Speaker Diarization systems
as specified in the evaluation ADR. Supports DER and JER calculations with RTTM format.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

# Import evaluation models
from eval.models import create_diarization_result

logger = logging.getLogger(__name__)


class RTTMSegment(NamedTuple):
    """Represents a segment from an RTTM file."""

    file_id: str
    channel: int
    start_time: float
    duration: float
    speaker_id: str
    confidence: float = 1.0


class DiarizationEvaluator:
    """Main evaluator class for Speaker Diarization systems.

    Provides methods to calculate DER and JER metrics following the NIST/DIHARD
    standards as defined in the evaluation ADR.
    """

    def __init__(self, collar: float = 0.25, skip_overlap: bool = False):
        """Initialize the diarization evaluator.

        Args:
            collar: Tolerance collar in seconds for timing errors
            skip_overlap: Whether to skip overlapping speech regions in evaluation
        """
        self.collar = collar
        self.skip_overlap = skip_overlap

    def calculate_der(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> dict[str, float]:
        """Calculate Diarization Error Rate.

        DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Speech

        Args:
            ground_truth_segments: Ground truth speaker segments
            generated_segments: Predicted speaker segments

        Returns
        -------
            Dictionary with DER components and total
        """
        if not ground_truth_segments:
            return {"der": 0.0, "missed_speech": 0.0, "false_alarm": 0.0, "speaker_confusion": 0.0}

        # Find the maximum time extent across both segmentations
        max_time = max(max((seg.start_time + seg.duration for seg in ground_truth_segments), default=0.0), max((seg.start_time + seg.duration for seg in generated_segments), default=0.0))

        # Create time-speaker matrices with same length
        ground_truth_matrix = self._segments_to_matrix(ground_truth_segments, max_end_time=max_time)
        generated_matrix = self._segments_to_matrix(generated_segments, max_end_time=max_time)

        # Apply collar if specified
        if self.collar > 0:
            ground_truth_matrix = self._apply_collar(ground_truth_matrix, self.collar)

        # Calculate DER components
        total_speech = np.sum(ground_truth_matrix > 0)

        if total_speech == 0:
            # No reference speech - check for false alarms
            false_alarms = np.sum(generated_matrix > 0)
            if false_alarms > 0:
                return {"der": 1.0, "missed_speech": 0.0, "false_alarm": 1.0, "speaker_confusion": 0.0}
            return {"der": 0.0, "missed_speech": 0.0, "false_alarm": 0.0, "speaker_confusion": 0.0}

        # Missed speech: ground truth has speech, generated doesn't
        missed_speech = np.sum((ground_truth_matrix > 0) & (generated_matrix == 0))

        # False alarm: generated has speech, ground truth doesn't
        false_alarm = np.sum((ground_truth_matrix == 0) & (generated_matrix > 0))

        # Speaker confusion: both have speech but different speakers
        confusion_mask = (ground_truth_matrix > 0) & (generated_matrix > 0) & (ground_truth_matrix != generated_matrix)
        speaker_confusion = np.sum(confusion_mask)

        # Calculate rates
        missed_rate = missed_speech / total_speech
        false_alarm_rate = false_alarm / total_speech
        confusion_rate = speaker_confusion / total_speech

        der = missed_rate + false_alarm_rate + confusion_rate

        return {"der": der, "missed_speech": missed_rate, "false_alarm": false_alarm_rate, "speaker_confusion": confusion_rate}

    def calculate_ser(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> float:
        """Calculate Segment Error Rate.

        SER is a segment-level error rate that handles arbitrary segmentation via
        connected sub-graphs and adaptive IoU threshold (following X-LANCE/BER).

        Args:
            ground_truth_segments: Ground truth speaker segments
            generated_segments: Predicted speaker segments

        Returns
        -------
            Segment Error Rate (between 0 and 1)
        """
        from scipy.optimize import linear_sum_assignment

        # Group segments by speaker
        ground_truth_speakers = self._group_segments_by_speaker(ground_truth_segments)
        generated_speakers = self._group_segments_by_speaker(generated_segments)

        if not ground_truth_speakers:
            return 0.0 if not generated_speakers else 1.0

        # Build speaker mapping using optimal assignment
        ref_speaker_list = list(ground_truth_speakers.keys())
        sys_speaker_list = list(generated_speakers.keys())

        # Build cost matrix for optimal speaker mapping
        cost_matrix = np.zeros((len(ref_speaker_list), len(sys_speaker_list)))
        for i, ref_spk in enumerate(ref_speaker_list):
            for j, sys_spk in enumerate(sys_speaker_list):
                # Cost is the intersection time between ref and sys speakers
                cost_matrix[i, j] = self._calculate_intervals_intersection(
                    [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_speakers[ref_spk]], [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]
                )

        # Optimal mapping
        ref_indices, sys_indices = linear_sum_assignment(-cost_matrix)

        total_ref_segments = sum(len(segs) for segs in ground_truth_speakers.values())
        error_segments = 0

        # Count segments in unmatched speakers
        matched_ref = set(ref_speaker_list[i] for i in ref_indices)

        for ref_spk in ground_truth_speakers:
            if ref_spk not in matched_ref:
                error_segments += len(ground_truth_speakers[ref_spk])
        # For matched speakers, check segment-level alignment using adaptive IoU
        for ref_idx, sys_idx in zip(ref_indices, sys_indices):
            ref_spk = ref_speaker_list[ref_idx]
            sys_spk = sys_speaker_list[sys_idx]

            ref_segs = [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_speakers[ref_spk]]
            sys_segs = [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]

            # Build connection matrix to find overlapping segments
            connection_matrix = self._build_connection_matrix(ref_segs, sys_segs)

            # Get connected sub-graphs
            connected_groups = self._get_connected_graphs(connection_matrix)

            # Track which ref segments have been processed
            processed_ref_ids = set()

            # Check each connected group with adaptive IoU
            for group in connected_groups:
                ref_seg_ids = set(i for i, j in group)
                sys_seg_ids = set(j for i, j in group)

                group_ref_segs = [ref_segs[i] for i in ref_seg_ids]
                group_sys_segs = [sys_segs[j] for j in sys_seg_ids]

                # Calculate IoU for this group
                intersection = self._calculate_intervals_intersection(group_ref_segs, group_sys_segs)
                union = self._calculate_intervals_union(group_ref_segs, group_sys_segs)

                if union > 0:
                    iou = intersection / union

                    # Adaptive IoU threshold based on segment count and duration
                    total_duration = sum(end - start for start, end in group_ref_segs)
                    collar = 0.5  # 0.25 * 2 (before and after boundary)
                    n_segs = len(group_ref_segs)
                    adaptive_iou = max(0.5, (total_duration - 2 * collar * n_segs) / (total_duration + 2 * collar * n_segs))

                    # If IoU below adaptive threshold, count as errors
                    if iou < adaptive_iou:
                        error_segments += len(ref_seg_ids)

                # Mark these segments as processed
                processed_ref_ids.update(ref_seg_ids)

            # Count isolated nodes (segments with no overlaps) - only those not already processed
            all_ref_ids = set(range(len(ref_segs)))
            isolated_ref = all_ref_ids - processed_ref_ids
            error_segments += len(isolated_ref)

        return float(error_segments / total_ref_segments) if total_ref_segments > 0 else 0.0

    def calculate_ber(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> dict[str, float]:
        """Calculate Balanced Error Rate.

        BER balances speaker-weighted error, duration error, and segment error.
        BER = ref_part + fa_mean, where:
        - ref_part: weighted average of (DER, SER) for each matched reference speaker
        - fa_mean: harmonic mean of FA duration and FA segments

        Args:
            ground_truth_segments: Ground truth speaker segments
            generated_segments: Predicted speaker segments

        Returns
        -------
            Dictionary with BER components
        """
        from scipy.optimize import linear_sum_assignment

        # Group segments by speaker
        ground_truth_speakers = self._group_segments_by_speaker(ground_truth_segments)
        generated_speakers = self._group_segments_by_speaker(generated_segments)

        if not ground_truth_speakers:
            return {"ber": 0.0, "ser": 0.0, "ref_part": 0.0, "fa_dur": 0.0, "fa_seg": 0.0, "fa_mean": 0.0}

        # Build optimal speaker mapping
        ref_speaker_list = list(ground_truth_speakers.keys())
        sys_speaker_list = list(generated_speakers.keys())

        cost_matrix = np.zeros((len(ref_speaker_list), len(sys_speaker_list)))
        for i, ref_spk in enumerate(ref_speaker_list):
            for j, sys_spk in enumerate(sys_speaker_list):
                cost_matrix[i, j] = self._calculate_intervals_intersection(
                    [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_speakers[ref_spk]], [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]
                )

        ref_indices, sys_indices = linear_sum_assignment(-cost_matrix)

        # Calculate per-speaker BER for matched speakers
        speaker_bers = []
        total_ref_time = 0
        total_ref_segments = 0
        total_fa_time = 0
        total_fa_segments = 0

        matched_sys = set(sys_speaker_list[j] for j in sys_indices)

        for ref_idx, sys_idx in zip(ref_indices, sys_indices):
            ref_spk = ref_speaker_list[ref_idx]
            sys_spk = sys_speaker_list[sys_idx]

            ref_segs = [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_speakers[ref_spk]]
            sys_segs = [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]

            # Duration-level error (similar to DER component)
            ref_duration = sum(end - start for start, end in ref_segs)
            total_ref_time += ref_duration

            # Calculate FA and MS durations
            fa_duration, ms_duration = self._calculate_fa_ms_duration(ref_segs, sys_segs)
            der = (fa_duration + ms_duration) / ref_duration if ref_duration > 0 else 0.0

            # Segment-level error for this speaker
            connection_matrix = self._build_connection_matrix(ref_segs, sys_segs)
            connected_groups = self._get_connected_graphs(connection_matrix)

            error_segs = 0
            for group in connected_groups:
                ref_seg_ids = set(i for i, j in group)
                sys_seg_ids = set(j for i, j in group)

                group_ref_segs = [ref_segs[i] for i in ref_seg_ids]
                group_sys_segs = [sys_segs[j] for j in sys_seg_ids]

                intersection = self._calculate_intervals_intersection(group_ref_segs, group_sys_segs)
                union = self._calculate_intervals_union(group_ref_segs, group_sys_segs)

                if union > 0:
                    iou = intersection / union
                    total_duration = sum(end - start for start, end in group_ref_segs)
                    n_segs = len(group_ref_segs)
                    adaptive_iou = max(0.5, (total_duration - n_segs) / (total_duration + n_segs))

                    if iou < adaptive_iou:
                        error_segs += len(ref_seg_ids)

            # Isolated nodes
            all_ref_ids = set(range(len(ref_segs)))
            connected_ref_ids = set(i for group in connected_groups for i, j in group)
            error_segs += len(all_ref_ids - connected_ref_ids)

            ser = error_segs / len(ref_segs) if len(ref_segs) > 0 else 0.0
            total_ref_segments += len(ref_segs)

            # Harmonic mean of DER and SER for this speaker
            eps = 1e-6
            speaker_ber = 2 / (1 / (der + eps) + 1 / (ser + eps)) - eps
            speaker_bers.append(speaker_ber)

        # False alarm speakers (unmatched system speakers)
        for sys_spk in generated_speakers:
            if sys_spk not in matched_sys:
                sys_segs = [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]
                total_fa_time += sum(end - start for start, end in sys_segs)
                total_fa_segments += len(sys_segs)

        # Calculate final BER
        ref_part = np.mean(speaker_bers) if speaker_bers else 0.0

        fa_dur = total_fa_time / total_ref_time if total_ref_time > 0 else 0.0
        fa_seg = total_fa_segments / total_ref_segments if total_ref_segments > 0 else 0.0

        eps = 1e-6
        fa_mean = 2 / (1 / (fa_dur + eps) + 1 / (fa_seg + eps)) - eps

        ber = ref_part + fa_mean
        ser = self.calculate_ser(ground_truth_segments, generated_segments)

        return {"ber": float(ber), "ser": float(ser), "ref_part": float(ref_part), "fa_dur": float(fa_dur), "fa_seg": float(fa_seg), "fa_mean": float(fa_mean)}

    def calculate_jer(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> float:
        """Calculate Jaccard Error Rate.

        JER = 1 - average Jaccard index over mapped speakers following dscore implementation

        Args:
            ground_truth_segments: Ground truth speaker segments
            generated_segments: Predicted speaker segments

        Returns
        -------
            Jaccard Error Rate (between 0 and 1)
        """
        # Group segments by speaker
        ground_truth_speakers = self._group_segments_by_speaker(ground_truth_segments)
        generated_speakers = self._group_segments_by_speaker(generated_segments)

        n_ref_speakers = len(ground_truth_speakers)
        n_sys_speakers = len(generated_speakers)

        # Handle edge cases
        if n_ref_speakers == 0 and n_sys_speakers > 0:
            # No reference speech, but system has speech - 100% error
            return 1.0
        elif n_ref_speakers > 0 and n_sys_speakers == 0:
            # Reference has speech, but system has none - 100% error
            return 1.0
        elif n_ref_speakers == 0 and n_sys_speakers == 0:
            # No speech in either - perfect match
            return 0.0

        # Calculate speaker durations and intersection matrix
        from scipy.optimize import linear_sum_assignment

        ref_speaker_list = list(ground_truth_speakers.keys())
        sys_speaker_list = list(generated_speakers.keys())

        # Calculate total duration for each speaker
        ref_durs = np.array([sum(seg.duration for seg in ground_truth_speakers[spk]) for spk in ref_speaker_list])
        sys_durs = np.array([sum(seg.duration for seg in generated_speakers[spk]) for spk in sys_speaker_list])

        # Build intersection matrix
        intersect_matrix = np.zeros((n_ref_speakers, n_sys_speakers))
        for i, ref_spk in enumerate(ref_speaker_list):
            for j, sys_spk in enumerate(sys_speaker_list):
                intersect_matrix[i, j] = self._calculate_intervals_intersection(
                    [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_speakers[ref_spk]], [(seg.start_time, seg.start_time + seg.duration) for seg in generated_speakers[sys_spk]]
                )

        # Calculate Jaccard error matrix: JER = (FA + MISS) / TOTAL
        # Following dscore: union = ref_dur + sys_dur - intersect
        ref_durs_matrix = np.tile(ref_durs, [n_sys_speakers, 1]).T
        sys_durs_matrix = np.tile(sys_durs, [n_ref_speakers, 1])
        union_matrix = ref_durs_matrix + sys_durs_matrix - intersect_matrix

        # JER per speaker pair
        with np.errstate(divide="ignore", invalid="ignore"):
            jer_matrix = 1.0 - (intersect_matrix / union_matrix)
            jer_matrix[union_matrix == 0] = 0.0  # If union is 0, both are 0, so JER is 0

        # Find optimal assignment using Hungarian algorithm
        ref_indices, sys_indices = linear_sum_assignment(jer_matrix)

        # Calculate average JER for assigned pairs
        jer_scores = [jer_matrix[i, j] for i, j in zip(ref_indices, sys_indices)]

        # Clip to valid range and return mean
        jer_scores = np.clip(jer_scores, 0.0, 1.0)
        return float(np.mean(jer_scores))

    def evaluate_full(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]):
        """Perform complete diarization evaluation.

        Parameters
        ----------
        ground_truth_segments : list[RTTMSegment]
            Ground truth speaker segments
        generated_segments : list[RTTMSegment]
            Predicted speaker segments

        Returns
        -------
        DiarizationEvaluationResult | dict[str, float]
            Structured evaluation results (Pydantic model if available, dict otherwise)

        Notes
        -----
        This method calculates DER, JER, SER, and BER metrics using the collar
        tolerance specified during evaluator initialization.
        """
        der_results = self.calculate_der(ground_truth_segments, generated_segments)
        jer = self.calculate_jer(ground_truth_segments, generated_segments)
        ser = self.calculate_ser(ground_truth_segments, generated_segments)
        ber_results = self.calculate_ber(ground_truth_segments, generated_segments)

        results = der_results.copy()
        results["jer"] = jer
        results["ser"] = ser
        results["ber"] = ber_results["ber"]
        results["ber_ref_part"] = ber_results["ref_part"]
        results["ber_fa_dur"] = ber_results["fa_dur"]
        results["ber_fa_seg"] = ber_results["fa_seg"]
        results["ber_fa_mean"] = ber_results["fa_mean"]

        # Calculate additional metadata for the model
        total_speech_time = sum(seg.duration for seg in ground_truth_segments)
        ground_truth_speakers_count = len(set(seg.speaker_id for seg in ground_truth_segments))
        generated_speakers_count = len(set(seg.speaker_id for seg in generated_segments))

        # Return structured Pydantic model
        return create_diarization_result(
            metrics=results,
            collar=self.collar,
            total_speech_time=total_speech_time,
            reference_speakers=ground_truth_speakers_count,
            hypothesis_speakers=generated_speakers_count,
        )

    def _segments_to_matrix(self, segments: list[RTTMSegment], resolution: float = 0.01, max_end_time: float | None = None) -> np.ndarray:
        """Convert segments to time-speaker matrix.

        Args:
            segments: List of RTTM segments
            resolution: Time resolution in seconds
            max_end_time: Maximum end time to consider (if None, computed from segments)

        Returns
        -------
            1D numpy array where each element is a speaker index (0 for silence)
        """
        if not segments:
            if max_end_time is None:
                return np.array([])
            n_frames = int(np.ceil(max_end_time / resolution))
            return np.zeros(n_frames, dtype=int)

        # Find time bounds
        if max_end_time is None:
            max_end_time = max(seg.start_time + seg.duration for seg in segments)
        n_frames = int(np.ceil(max_end_time / resolution))

        # Create speaker ID mapping
        speaker_ids = sorted(set(seg.speaker_id for seg in segments))
        speaker_to_idx = {spk: idx + 1 for idx, spk in enumerate(speaker_ids)}  # +1 to keep 0 as silence

        # Initialize matrix
        matrix = np.zeros(n_frames, dtype=int)

        # Fill matrix with speaker assignments
        for segment in segments:
            start_frame = int(segment.start_time / resolution)
            end_frame = int((segment.start_time + segment.duration) / resolution)
            end_frame = min(end_frame, n_frames)  # Ensure we don't exceed bounds
            speaker_idx = speaker_to_idx[segment.speaker_id]

            matrix[start_frame:end_frame] = speaker_idx

        return matrix

    def _apply_collar(self, matrix: np.ndarray, collar: float, resolution: float = 0.01) -> np.ndarray:
        """Apply collar around speaker boundaries.

        Args:
            matrix: Time-speaker matrix
            collar: Collar size in seconds
            resolution: Time resolution in seconds

        Returns
        -------
            Matrix with collar applied
        """
        collar_frames = int(collar / resolution)

        # Find speaker change points
        diff = np.diff(matrix)
        change_points = np.where(diff != 0)[0]

        # Apply collar around change points
        collared_matrix = matrix.copy()

        for cp in change_points:
            start = max(0, cp - collar_frames)
            end = min(len(matrix), cp + collar_frames + 1)
            collared_matrix[start:end] = 0  # Set to silence in collar region

        return collared_matrix

    def _group_segments_by_speaker(self, segments: list[RTTMSegment]) -> dict[str, list[RTTMSegment]]:
        """Group segments by speaker ID."""
        speakers = {}
        for segment in segments:
            if segment.speaker_id not in speakers:
                speakers[segment.speaker_id] = []
            speakers[segment.speaker_id].append(segment)
        return speakers

    def _calculate_jaccard_for_speakers(self, ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> float:
        """Calculate Jaccard index between two sets of segments."""
        # Convert to time intervals
        ground_truth_intervals = [(seg.start_time, seg.start_time + seg.duration) for seg in ground_truth_segments]
        generated_intervals = [(seg.start_time, seg.start_time + seg.duration) for seg in generated_segments]

        # Calculate intersection and union
        intersection = self._calculate_intervals_intersection(ground_truth_intervals, generated_intervals)
        union = self._calculate_intervals_union(ground_truth_intervals, generated_intervals)

        if union == 0:
            return 1.0 if intersection == 0 else 0.0

        return intersection / union

    def _calculate_intervals_intersection(self, intervals1: list[tuple[float, float]], intervals2: list[tuple[float, float]]) -> float:
        """Calculate total intersection time between two sets of intervals."""
        total_intersection = 0.0

        for start1, end1 in intervals1:
            for start2, end2 in intervals2:
                intersection_start = max(start1, start2)
                intersection_end = min(end1, end2)

                if intersection_start < intersection_end:
                    total_intersection += intersection_end - intersection_start

        return total_intersection

    def _calculate_intervals_union(self, intervals1: list[tuple[float, float]], intervals2: list[tuple[float, float]]) -> float:
        """Calculate total union time between two sets of intervals."""
        # Merge all intervals and calculate total coverage
        all_intervals = intervals1 + intervals2

        if not all_intervals:
            return 0.0

        # Sort intervals by start time
        all_intervals.sort()

        # Merge overlapping intervals
        merged = [all_intervals[0]]

        for start, end in all_intervals[1:]:
            last_start, last_end = merged[-1]

            if start <= last_end:
                # Overlapping or adjacent, merge
                merged[-1] = (last_start, max(last_end, end))
            else:
                # Non-overlapping, add new interval
                merged.append((start, end))

        # Calculate total duration
        return sum(end - start for start, end in merged)

    def _build_connection_matrix(self, ref_segments: list[tuple[float, float]], hyp_segments: list[tuple[float, float]]) -> np.ndarray:
        """Build connection matrix indicating overlapping segments.

        Args:
            ref_segments: List of (start, end) tuples for reference
            hyp_segments: List of (start, end) tuples for hypothesis

        Returns
        -------
            2D binary array where [i,j]=1 means segments i and j overlap
        """
        connection_matrix = np.zeros((len(ref_segments), len(hyp_segments)), dtype=int)

        for i, (ref_start, ref_end) in enumerate(ref_segments):
            for j, (hyp_start, hyp_end) in enumerate(hyp_segments):
                max_start = max(ref_start, hyp_start)
                min_end = min(ref_end, hyp_end)
                if max_start < min_end:  # Overlapping
                    connection_matrix[i, j] = 1

        return connection_matrix

    def _get_connected_graphs(self, connection_matrix: np.ndarray) -> list[list[tuple[int, int]]]:
        """Get all connected sub-graphs using bipartite graph connectivity.

        In a bipartite graph (ref segments vs sys segments):
        - Two ref segments are in the same group if they connect to a common sys segment
        - Two sys segments are in the same group if they connect to a common ref segment

        Args:
            connection_matrix: Binary matrix indicating connections [n_ref x n_sys]

        Returns
        -------
            List of connected groups, each group is a list of (i,j) pairs
        """
        n_ref, n_sys = connection_matrix.shape

        # Track which ref and sys nodes have been assigned to a group
        ref_assigned = np.zeros(n_ref, dtype=bool)
        sys_assigned = np.zeros(n_sys, dtype=bool)

        groups = []

        def expand_group(ref_nodes: set[int], sys_nodes: set[int]):
            """Expand a group by finding all connected nodes."""
            changed = True
            while changed:
                changed = False
                # Find sys nodes connected to current ref nodes
                for i in ref_nodes:
                    for j in range(n_sys):
                        if connection_matrix[i, j] == 1 and j not in sys_nodes:
                            sys_nodes.add(j)
                            changed = True

                # Find ref nodes connected to current sys nodes
                for j in sys_nodes:
                    for i in range(n_ref):
                        if connection_matrix[i, j] == 1 and i not in ref_nodes:
                            ref_nodes.add(i)
                            changed = True

        # Find all connected components
        for i in range(n_ref):
            if not ref_assigned[i]:
                # Check if this ref node has any connections
                if np.any(connection_matrix[i, :] == 1):
                    # Start a new group with this ref node
                    ref_nodes = {i}
                    sys_nodes = set()

                    # Expand to find all connected nodes
                    expand_group(ref_nodes, sys_nodes)

                    # Mark as assigned
                    for r in ref_nodes:
                        ref_assigned[r] = True
                    for s in sys_nodes:
                        sys_assigned[s] = True

                    # Create group with all (i,j) pairs
                    group = [(r, s) for r in ref_nodes for s in sys_nodes if connection_matrix[r, s] == 1]
                    groups.append(group)

        return groups

    def _calculate_fa_ms_duration(self, ref_segments: list[tuple[float, float]], hyp_segments: list[tuple[float, float]], precision: int = 100) -> tuple[float, float]:
        """Calculate false alarm and missed speech duration.

        Args:
            ref_segments: Reference segments as (start, end) tuples
            hyp_segments: Hypothesis segments as (start, end) tuples
            precision: Time resolution (100 = 0.01s resolution)

        Returns
        -------
            Tuple of (fa_duration, ms_duration) in seconds
        """
        if not ref_segments and not hyp_segments:
            return 0.0, 0.0

        if not ref_segments:
            total_hyp = sum(end - start for start, end in hyp_segments)
            return total_hyp, 0.0

        if not hyp_segments:
            total_ref = sum(end - start for start, end in ref_segments)
            return 0.0, total_ref

        # Find maximum time extent
        max_time = max(max(end for start, end in ref_segments), max(end for start, end in hyp_segments))

        # Create boolean vectors
        n_frames = int(np.ceil(max_time * precision)) + 1
        ref_vector = np.zeros(n_frames, dtype=bool)
        hyp_vector = np.zeros(n_frames, dtype=bool)

        # Fill reference vector
        for start, end in ref_segments:
            start_idx = int(np.round(start * precision))
            end_idx = int(np.round(end * precision))
            ref_vector[start_idx:end_idx] = True

        # Fill hypothesis vector
        for start, end in hyp_segments:
            start_idx = int(np.round(start * precision))
            end_idx = int(np.round(end * precision))
            hyp_vector[start_idx:end_idx] = True

        # Calculate FA and MS
        fa_frames = np.sum(~ref_vector & hyp_vector)
        ms_frames = np.sum(ref_vector & ~hyp_vector)

        fa_duration = fa_frames / precision
        ms_duration = ms_frames / precision

        return fa_duration, ms_duration


def load_rttm_file(file_path: str | Path) -> list[RTTMSegment]:
    """Load RTTM file and parse segments.

    Args:
        file_path: Path to RTTM file

    Returns
    -------
        List of RTTMSegment objects
    """
    segments = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                segment = parse_rttm_line(line)
                if segment:
                    segments.append(segment)

    return segments


def parse_rttm_line(line: str) -> RTTMSegment | None:
    """Parse a single line from an RTTM file.

    RTTM format: SPEAKER <file-id> <channel> <start-time> <duration> <ortho> <stype> <name> <conf>

    Args:
        line: RTTM line to parse

    Returns
    -------
        RTTMSegment object or None if parsing fails
    """
    parts = line.strip().split()

    if len(parts) < 8 or parts[0] != "SPEAKER":
        return None

    try:
        file_id = parts[1]
        channel = int(parts[2])
        start_time = float(parts[3])
        duration = float(parts[4])
        speaker_id = parts[7]

        # Confidence is optional
        confidence = 1.0
        # float(parts[8]) if len(parts) > 8 else 1.0

        return RTTMSegment(file_id=file_id, channel=channel, start_time=start_time, duration=duration, speaker_id=speaker_id, confidence=confidence)

    except (ValueError, IndexError):
        return None


# Convenience functions for direct metric calculation
def calculate_der(ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment], collar: float = 0.25) -> dict[str, float]:
    """Calculate Diarization Error Rate."""
    evaluator = DiarizationEvaluator(collar=collar)
    return evaluator.calculate_der(ground_truth_segments, generated_segments)


def calculate_jer(ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment]) -> float:
    """Calculate Jaccard Error Rate."""
    evaluator = DiarizationEvaluator()
    return evaluator.calculate_jer(ground_truth_segments, generated_segments)


def evaluate_diarization_model(ground_truth_segments: list[RTTMSegment], generated_segments: list[RTTMSegment], collar: float = 0.25):
    """Comprehensive diarization model evaluation.

    Parameters
    ----------
    ground_truth_segments : list[RTTMSegment]
        Ground truth speaker segments from RTTM file
    generated_segments : list[RTTMSegment]
        Predicted speaker segments from RTTM file
    collar : float, optional
        Tolerance collar in seconds for timing errors, by default 0.25

    Returns
    -------
    DiarizationEvaluationResult | dict[str, float]
        Structured evaluation results (Pydantic model if available, dict otherwise)

    Examples
    --------
    >>> ground_truth_segs = [RTTMSegment("file1", 1, 0.0, 2.0, "speaker1")]
    >>> generated_segs = [RTTMSegment("file1", 1, 0.1, 1.9, "speaker1")]
    >>> results = evaluate_diarization_model(ground_truth_segs, generated_segs, collar=0.25)
    >>> print(f"DER: {results.der:.4f}" if hasattr(results, 'der') else f"DER: {results['der']:.4f}")
    """
    evaluator = DiarizationEvaluator(collar=collar)
    return evaluator.evaluate_full(ground_truth_segments, generated_segments)


if __name__ == "__main__":
    # Example usage

    # Create sample segments
    ground_truth_segments = [
        RTTMSegment("file1", 1, 0.0, 2.0, "speaker1"),
        RTTMSegment("file1", 1, 2.5, 1.5, "speaker2"),
        RTTMSegment("file1", 1, 4.5, 2.0, "speaker1"),
    ]

    generated_segments = [
        RTTMSegment("file1", 1, 0.1, 1.8, "spk_A"),
        RTTMSegment("file1", 1, 2.6, 1.4, "spk_B"),
        RTTMSegment("file1", 1, 4.6, 1.9, "spk_A"),
    ]

    evaluator = DiarizationEvaluator()
    results = evaluator.evaluate_full(ground_truth_segments, generated_segments)

    logger.info("Diarization Evaluation Results (with ADR-003 Rubric Scores):")
    logger.info(f"DER: {results.der:.4f} -> {results.der_rubric.rubric_score}/4 ({results.der_rubric.rubric_label})")
    logger.info(f"JER: {results.jer:.4f} -> {results.jer_rubric.rubric_score}/4 ({results.jer_rubric.rubric_label})")
    logger.info(f"Missed Speech: {results.missed_speech:.4f} -> {results.missed_speech_rubric.rubric_score}/4 ({results.missed_speech_rubric.rubric_label})")
    logger.info(f"False Alarm: {results.false_alarm:.4f} -> {results.false_alarm_rubric.rubric_score}/4 ({results.false_alarm_rubric.rubric_label})")
    logger.info(f"Speaker Confusion: {results.speaker_confusion:.4f} -> {results.speaker_confusion_rubric.rubric_score}/4 ({results.speaker_confusion_rubric.rubric_label})")
    logger.info(f"Collar: {results.collar:.4f}s")
