import cv2
import numpy as np
from typing import List
from collections import defaultdict
from sklearn.cluster import KMeans
from .base_team_assigner import BaseTeamClassifier


def _bgr_to_color_name(bgr: np.ndarray) -> str:
    """Map a BGR color vector to a human-readable color name."""
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 50:
        return "black"
    if s < 40:
        if v > 180:
            return "white"
        if v > 100:
            return "gray"
        return "dark"
    # Chromatic colors
    if h < 10 or h > 170:
        return "red"
    if h < 25:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


class BasketballTeamClassifier(BaseTeamClassifier):
    """
    Basketball team classifier using zero-shot text-prompt classification.

    Instead of unsupervised clustering (which separates by pose, not jersey
    color), this classifier:
      1. Auto-detects the two dominant jersey colors from sampled crops.
      2. Builds text prompts like "a person wearing a red jersey".
      3. Uses SigLIP2's text+vision encoder to classify each crop.
    """

    def __init__(self, **kwargs):
        # Force text encoder to be loaded for zero-shot classification
        kwargs['use_text_encoder'] = True
        super().__init__(**kwargs)
        self.team_labels = []

    def _get_expected_clusters(self) -> int:
        # Not used in zero-shot mode, but required by base class interface
        return 2

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        h, w, _ = crop.shape
        if h < 10 or w < 10:
            return crop
        # Target only the torso/tank-top to avoid court and leg/skin color
        return crop[int(h * 0.15):int(h * 0.65), int(w * 0.3):int(w * 0.7), :]

    def _map_clusters_to_teams(self, unique, counts, N):
        pass  # Not used in zero-shot mode

    def _get_raw_tids(self, evidence, frame_t0_xys, frame_t1_xys):
        pass  # Not used in zero-shot mode

    # ------ Auto-detection of jersey colors ------

    @staticmethod
    def _collect_bgr_stats(
        tracks_players, frame_generator, sample_stride: int, preprocess_fn
    ) -> np.ndarray:
        """Collect mean BGR stats from torso crops across sampled frames."""
        all_bgr = []
        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= len(tracks_players):
                break
            if frame_num % sample_stride != 0:
                continue
            h_f, w_f = frame.shape[:2]
            for track in tracks_players[frame_num].values():
                x1, y1, x2, y2 = map(int, track['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < 400:
                    continue
                crop = frame[y1:y2, x1:x2]
                torso = preprocess_fn(crop)
                if torso.size == 0:
                    continue
                # Filter very dark crops (likely referees)
                mean_v = np.mean(torso)
                if mean_v < 50:
                    continue
                mean_bgr = np.mean(torso, axis=(0, 1))
                all_bgr.append(mean_bgr)
        return np.array(all_bgr) if all_bgr else np.empty((0, 3))

    def _detect_jersey_colors(self, bgr_stats: np.ndarray) -> List[str]:
        """Run KMeans(k=2) on BGR stats to find the two dominant jersey colors."""
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        km.fit(bgr_stats)
        centers = km.cluster_centers_

        color_names = [_bgr_to_color_name(c) for c in centers]

        # If both clusters map to the same name, differentiate by brightness
        if color_names[0] == color_names[1]:
            v0 = np.mean(centers[0])
            v1 = np.mean(centers[1])
            if v0 > v1:
                color_names[0] = "light " + color_names[0]
                color_names[1] = "dark " + color_names[1]
            else:
                color_names[0] = "dark " + color_names[0]
                color_names[1] = "light " + color_names[1]

        return color_names

    # ------ Overridden pipeline methods ------

    # Candidate prompts for auto-detecting jersey colors via zero-shot
    _CANDIDATE_PROMPTS = [
        "a basketball player wearing a red jersey",
        "a basketball player wearing a white jersey",
        "a basketball player wearing a blue jersey",
        "a basketball player wearing a black jersey",
        "a basketball player wearing a yellow jersey",
        "a basketball player wearing a green jersey",
        "a basketball player wearing a purple jersey",
        "a basketball player wearing an orange jersey",
        "a basketball player wearing a gray jersey",
    ]

    def fit_from_video(
        self, tracks_players, frame_generator, sample_stride: int = 10
    ) -> None:
        """Auto-detect jersey colors using SigLIP zero-shot, then set final prompts."""
        from collections import Counter

        print("  [Basketball] Auto-detecting jersey colors via zero-shot...")

        # Step 1: Pre-compute embeddings for ALL candidate prompts
        self.precompute_text_embeddings(self._CANDIDATE_PROMPTS)

        # Step 2: Collect sample crops from the video
        crops = []
        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= len(tracks_players):
                break
            if frame_num % sample_stride != 0:
                continue
            h_f, w_f = frame.shape[:2]
            for track in tracks_players[frame_num].values():
                x1, y1, x2, y2 = map(int, track['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < 400:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(crop)

        if not crops:
            print("  ⚠ No crops collected! Defaulting to red/white.")
            self.team_labels = [
                "a basketball player wearing a red jersey",
                "a basketball player wearing a white jersey",
            ]
            self.precompute_text_embeddings(self.team_labels)
            return

        # Step 3: Classify all sample crops against ALL candidates
        predictions = self.zero_shot_classify(crops)

        # Step 4: Count how often each candidate wins
        vote_counts = Counter(int(p) for p in predictions)
        print(f"    Vote distribution across {len(crops)} crops:")
        for idx, count in vote_counts.most_common():
            print(f"      {self._CANDIDATE_PROMPTS[idx]}: {count}")

        # Step 5: Pick the top 2 most common labels as the two teams
        top_two = vote_counts.most_common(2)
        if len(top_two) < 2:
            print("  ⚠ Could not find 2 distinct teams! Defaulting to red/white.")
            self.team_labels = [
                "a basketball player wearing a red jersey",
                "a basketball player wearing a white jersey",
            ]
        else:
            self.team_labels = [
                self._CANDIDATE_PROMPTS[top_two[0][0]],
                self._CANDIDATE_PROMPTS[top_two[1][0]],
            ]

        print(f"    → Final team prompts: {self.team_labels}")

        # Step 6: Re-compute text embeddings for just the final 2 labels
        self.precompute_text_embeddings(self.team_labels)

    def classify_from_video(
        self, tracks, frame_generator, team_colors, sample_stride: int = 5
    ) -> None:
        """Classify each player crop via zero-shot text similarity."""
        total = len(tracks['players'])
        print(f"  [Zero-Shot] Classifying {total} frames (stride={sample_stride})...")

        # Pass 1: collect crops and classify
        track_evidence = defaultdict(list)

        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= total:
                break
            if frame_num % sample_stride != 0:
                continue

            h_f, w_f = frame.shape[:2]
            crops, pids = [], []

            for pid, info in tracks['players'][frame_num].items():
                x1, y1, x2, y2 = map(int, info['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < 400:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append(crop)
                pids.append(pid)

            if not crops:
                continue

            # Zero-shot classify this batch of crops
            predictions = self.zero_shot_classify(crops)

            for pid, pred in zip(pids, predictions):
                track_evidence[pid].append({
                    "frame": frame_num,
                    "tid": int(pred),
                })

            if (frame_num + 1) % 200 == 0:
                print(f"    Frame {frame_num + 1}/{total}")

        # Pass 2: aggregate per-track evidence with majority-vote smoothing
        print(f"  [Zero-Shot] Aggregating evidence for {len(track_evidence)} tracks...")
        final_assignments = {}

        for pid, evidence in track_evidence.items():
            evidence.sort(key=lambda x: x["frame"])
            known_frames = [e["frame"] for e in evidence]
            tids = [e["tid"] for e in evidence]

            if not known_frames:
                continue

            # Temporal smoothing: sliding window majority vote (window=9)
            smoothed = {}
            for i, f in enumerate(known_frames):
                start_i = max(0, i - 4)
                end_i = min(len(tids), i + 5)
                window = tids[start_i:end_i]
                smoothed[f] = 0 if window.count(0) > window.count(1) else 1

            # Propagate to all frames
            final_assignments[pid] = {}
            first_f, last_f = known_frames[0], known_frames[-1]

            for f in range(0, first_f):
                final_assignments[pid][f] = smoothed[first_f]
            for f in range(last_f + 1, total):
                final_assignments[pid][f] = smoothed[last_f]

            current_tid = smoothed[first_f]
            for f in range(first_f, last_f + 1):
                if f in smoothed:
                    current_tid = smoothed[f]
                final_assignments[pid][f] = current_tid

        # Pass 3: write assignments
        print("  [Zero-Shot] Writing assignments...")
        self._apply_assignments(tracks, final_assignments, team_colors)
        print(f"  Done. Resolved {len(final_assignments)} unique players.")

    def release_model(self):
        """Release both the full model and vision model."""
        import gc
        import torch
        if hasattr(self, 'full_model') and self.full_model is not None:
            del self.full_model
        if hasattr(self, 'features_model'):
            del self.features_model
        if hasattr(self, 'processor'):
            del self.processor
        self._text_embeddings = None
        gc.collect()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        print(f"  [Memory] Released {self.model_type} + CUDA cache cleared.")
