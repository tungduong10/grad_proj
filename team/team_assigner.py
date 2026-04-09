import os
import gc
from collections import Counter, defaultdict
from typing import Generator, Iterable, List, TypeVar

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPVisionModelWithProjection

V = TypeVar("V")

FASHIONCLIP_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'model', 'fashion-clip'
)


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    Memory- and speed-optimised team classifier.

    Uses FashionCLIP (ViT-B/32) for feature extraction — significantly
    faster on CPU than SigLIP (ViT-B/16).

    Strategy:
      1. fit_from_video      – sample a subset of frames → extract embeddings
                               → fit UMAP + KMeans.
      2. classify_from_video – stream through sampled frames, run inference
                               only on those, cache team-ID per tracker-ID.
                               Non-sampled frames propagate cached IDs.
      3. After classification, the model can be released.
    """

    def __init__(self, device: str = 'cpu', batch_size: int = 32, 
                 model_path: str = None, use_fp16: bool = False):
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        
        path = model_path if model_path else FASHIONCLIP_MODEL_PATH

        # FashionCLIP vision-only (ViT-B/32, ~350M → much lighter than SigLIP)
        self.features_model = CLIPVisionModelWithProjection.from_pretrained(
            path
        ).to(device)
        
        if self.use_fp16 and 'cuda' in str(device):
            self.features_model = self.features_model.half()
            
        self.features_model.eval()
        self.processor = CLIPProcessor.from_pretrained(path)

        self.reducer = umap.UMAP(n_components=3, random_state=42)
        # 5 clusters: Team 0, Team 1, GK 1, GK 2, and Noise/Crowd
        self.cluster_model = KMeans(n_clusters=5, random_state=42)

        self.team_0_cluster = None
        self.team_1_cluster = None
        self.gk_clusters = []
        self.team_fingerprints = {} # cluster_id -> np.array([H, S, V])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
        """Torso-center focus: Focuses on jersey while removing head, boots, and background."""
        h, w, _ = crop.shape
        if h < 10 or w < 10:
            return crop
        # Center-torso: 20% to 50% height, 20% to 80% width
        return crop[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8), :]

    @staticmethod
    def _get_color_stats(crop: np.ndarray) -> np.ndarray:
        """Calculate the average [H, S, V] of the torso crop."""
        if crop.size == 0:
            return np.array([0, 0, 0])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return np.mean(hsv, axis=(0, 1))

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Run FashionCLIP vision encoder on a list of crops in small batches."""
        # Preprocess and resize immediately to save RAM
        processed_crops = []
        for c in crops:
            crop = self._preprocess_crop(c)
            if crop.size > 0:
                # Resize to standard CLIP input size to save memory
                crop = cv2.resize(crop, (224, 224))
                processed_crops.append(sv.cv2_to_pillow(crop))
        
        data = []
        with torch.no_grad():
            for batch in create_batches(processed_crops, self.batch_size):
                inputs = self.processor(
                    images=batch, return_tensors="pt"
                ).to(self.device)
                
                if self.use_fp16 and 'cuda' in str(self.device):
                    # inputs['pixel_values'] is what the model takes
                    inputs['pixel_values'] = inputs['pixel_values'].half()
                
                outputs = self.features_model(**inputs)
                # Mean-pool last_hidden_state for richer visual features
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(emb)
                del inputs, outputs, emb
        del processed_crops
        gc.collect()
        return np.concatenate(data)

    def _resolve_gk(self, cluster_ids: np.ndarray,
                    players_xy: np.ndarray) -> np.ndarray:
        """Map raw cluster IDs → team 0/1, resolving GK by proximity."""
        team_0_mask = cluster_ids == self.team_0_cluster
        team_1_mask = cluster_ids == self.team_1_cluster

        t0_cent = (players_xy[team_0_mask].mean(axis=0)
                   if np.any(team_0_mask) else np.array([0, 0]))
        t1_cent = (players_xy[team_1_mask].mean(axis=0)
                   if np.any(team_1_mask) else np.array([0, 0]))

        out = np.empty(len(cluster_ids), dtype=int)
        for i, cid in enumerate(cluster_ids):
            if cid == self.team_0_cluster:
                out[i] = 0
            elif cid == self.team_1_cluster:
                out[i] = 1
            else:
                d0 = np.linalg.norm(players_xy[i] - t0_cent)
                d1 = np.linalg.norm(players_xy[i] - t1_cent)
                out[i] = 0 if d0 < d1 else 1
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_from_video(self, tracks_players, frame_generator,
                       sample_stride: int = 30) -> None:
        """
        Fit UMAP + KMeans by sampling every `sample_stride`-th frame.
        """
        all_embeddings = []
        all_hsv_stats = []
        sampled = 0

        for frame_num, frame in enumerate(frame_generator):
            if frame_num % sample_stride != 0:
                continue
            if frame_num >= len(tracks_players):
                break

            crops = []
            for track in tracks_players[frame_num].values():
                x1, y1, x2, y2 = map(int, track['bbox'])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

            if crops:
                processed = [self._preprocess_crop(c) for c in crops]
                all_hsv_stats.append(np.array([self._get_color_stats(pc) for pc in processed]))
                all_embeddings.append(self.extract_features(crops))
                sampled += len(crops)

        if not all_embeddings:
            print("⚠ No crops collected for fitting!")
            return

        all_embeddings = np.concatenate(all_embeddings)
        all_hsv_stats = np.concatenate(all_hsv_stats)
        print(f"  Fitting on {len(all_embeddings)} sampled crops "
              f"(stride={sample_stride})...")

        projections = self.reducer.fit_transform(all_embeddings)
        self.cluster_model.fit(projections)

        labels = self.cluster_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]

        self.team_0_cluster = sorted_clusters[0]
        self.team_1_cluster = sorted_clusters[1]
        self.gk_clusters = sorted_clusters[2:].tolist()

        # Learn fingerprints for each cluster
        self.team_fingerprints = {}
        for cid in unique:
            mask = labels == cid
            self.team_fingerprints[int(cid)] = np.mean(all_hsv_stats[mask], axis=0)
            stats = self.team_fingerprints[int(cid)]
            print(f"    - Cluster {cid} Profile: H={stats[0]:.1f}, S={stats[1]:.1f}, V={stats[2]:.1f}")

        del all_embeddings, projections, all_hsv_stats
        gc.collect()

    def classify_from_video(self, tracks, frame_generator, team_colors,
                            sample_stride: int = 10) -> None:
        """
        Assign teams using Adaptive Fingerprinting Resolution.
        """
        # pid -> list of {"cid": cluster_id, "xy": [x, y], "hsv": [h, s, v]}
        track_evidence = defaultdict(list)
        total = len(tracks['players'])

        # Pass 1: Video sweep to collect visual evidence
        print(f"  [Team Assignment] Pass 1: Collecting evidence (stride={sample_stride})...")
        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= total:
                break
            
            if frame_num % sample_stride != 0:
                continue

            player_track = tracks['players'][frame_num]
            crops, xys, pids = [], [], []
            for pid, info in player_track.items():
                x1, y1, x2, y2 = map(int, info['bbox'])
                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size == 0:
                    continue
                crops.append(raw_crop)
                xys.append([(x1 + x2) / 2, y2])
                pids.append(pid)

            if crops:
                torso_crops = [self._preprocess_crop(c) for c in crops]
                hsv_stats = [self._get_color_stats(tc) for tc in torso_crops]
                
                emb = self.extract_features(crops)
                proj = self.reducer.transform(emb)
                cids = self.cluster_model.predict(proj)
                
                for pid, cid, xy, hsv in zip(pids, cids, xys, hsv_stats):
                    track_evidence[pid].append({
                        "cid": int(cid), 
                        "xy": xy, 
                        "hsv": hsv
                    })
                del emb, proj, cids

            if (frame_num + 1) % 100 == 0 or (frame_num + 1) == total:
                print(f"    - Frame {frame_num + 1} / {total}")

        # Pass 2: Global resolution with Adaptive Fingerprinting
        print("  [Team Assignment] Pass 2: Resolving track-level assignments with Adaptive Fingerprinting...")
        
        # Team profiles (Learned during Pass 1 sweep for all detections)
        # We focus on Team 0 and Team 1 cluster profiles
        t0_prof = self.team_fingerprints.get(self.team_0_cluster, np.array([0, 0, 0]))
        t1_prof = self.team_fingerprints.get(self.team_1_cluster, np.array([0, 0, 0]))

        # Global centroids for GK resolution
        all_t0_xys = [ev['xy'] for pid in track_evidence for ev in track_evidence[pid] if ev['cid'] == self.team_0_cluster]
        all_t1_xys = [ev['xy'] for pid in track_evidence for ev in track_evidence[pid] if ev['cid'] == self.team_1_cluster]
        
        t0_global_cent = np.mean(all_t0_xys, axis=0) if all_t0_xys else np.array([0, 0])
        t1_global_cent = np.mean(all_t1_xys, axis=0) if all_t1_xys else np.array([0, 0])

        final_assignments = {} # pid -> team_id
        for pid, evidence in track_evidence.items():
            cids = [ev['cid'] for ev in evidence]
            avg_hsv = np.mean([ev['hsv'] for ev in evidence], axis=0)
            majority_cid = Counter(cids).most_common(1)[0][0]

            # --- ADAPTIVE COLOR GUARDRAIL ---
            # Calculate distance to learned fingerprints in HSV space
            # Weighting S and V more heavily for jersey color differentiation
            def color_dist(hsv1, hsv2):
                # Hue is circular (0-180 in OpenCV)
                dh = min(abs(hsv1[0] - hsv2[0]), 180 - abs(hsv1[0] - hsv2[0])) / 180.0
                ds = abs(hsv1[1] - hsv2[1]) / 255.0
                dv = abs(hsv1[2] - hsv2[2]) / 255.0
                return np.sqrt(dh*dh + ds*ds + dv*dv)

            d0 = color_dist(avg_hsv, t0_prof)
            d1 = color_dist(avg_hsv, t1_prof)

            # If the track's color profile is clearly closer to the other team's fingerprint
            # than its assigned cluster, we override.
            if d1 < d0 * 0.7: # Significant preference for Team 1
                final_assignments[pid] = 1
            elif d0 < d1 * 0.7: # Significant preference for Team 0
                final_assignments[pid] = 0
            else:
                # Fallback to KMeans Majority cluster
                if majority_cid == self.team_0_cluster:
                    final_assignments[pid] = 0
                elif majority_cid == self.team_1_cluster:
                    final_assignments[pid] = 1
                else:
                    # GK resolution using positional centroids
                    avg_xy = np.mean([ev['xy'] for ev in evidence], axis=0)
                    dist0 = np.linalg.norm(avg_xy - t0_global_cent)
                    dist1 = np.linalg.norm(avg_xy - t1_global_cent)
                    final_assignments[pid] = 0 if dist0 < dist1 else 1

        # Pass 3: Apply to all frames
        print("  [Team Assignment] Pass 3: Applying fixed assignments to tracks...")
        for frame_num in range(total):
            for pid, info in tracks['players'][frame_num].items():
                tid = final_assignments.get(pid, 0) # Default to 0
                tracks['players'][frame_num][pid]['team'] = tid
                tracks['players'][frame_num][pid]['team_color'] = \
                    team_colors.get(tid, (255, 255, 255))
        
        print(f"  [Team Assignment] Done. Resolved {len(final_assignments)} unique players.")

    def release_model(self):
        """Free the FashionCLIP model from memory."""
        del self.features_model
        del self.processor
        gc.collect()
        print("  [Memory] Released FashionCLIP model.")

    # Legacy wrappers (kept for compatibility) -------------------------
    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        labels = self.cluster_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]
        self.team_0_cluster = sorted_clusters[0]
        self.team_1_cluster = sorted_clusters[1]
        self.gk_clusters = sorted_clusters[2:].tolist()

    def predict(self, crops, players_xy):
        return self.predict_frame(crops, players_xy)

    def predict_frame(self, frame_crops, players_xy):
        if len(frame_crops) == 0:
            return np.array([])
        data = self.extract_features(frame_crops)
        proj = self.reducer.transform(data)
        cids = self.cluster_model.predict(proj)
        return self._resolve_gk(cids, players_xy)