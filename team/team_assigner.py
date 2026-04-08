import os
import gc
from typing import Generator, Iterable, List, TypeVar

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

    def __init__(self, device: str = 'cpu', batch_size: int = 16):
        self.device = device
        self.batch_size = batch_size

        # FashionCLIP vision-only (ViT-B/32, ~350M → much lighter than SigLIP)
        self.features_model = CLIPVisionModelWithProjection.from_pretrained(
            FASHIONCLIP_MODEL_PATH
        ).to(device)
        self.features_model.eval()
        self.processor = CLIPProcessor.from_pretrained(FASHIONCLIP_MODEL_PATH)

        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=4)

        self.team_0_cluster = None
        self.team_1_cluster = None
        self.gk_clusters = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
        """Upper half only → removes grass background."""
        return crop[:crop.shape[0] // 2, :, :]

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Run FashionCLIP vision encoder on a list of crops in small batches."""
        crops = [self._preprocess_crop(c) for c in crops]
        pil_crops = [sv.cv2_to_pillow(c) for c in crops]
        data = []
        with torch.no_grad():
            for batch in create_batches(pil_crops, self.batch_size):
                inputs = self.processor(
                    images=batch, return_tensors="pt"
                ).to(self.device)
                outputs = self.features_model(**inputs)
                # Mean-pool last_hidden_state for richer visual features
                # (better for jersey color clustering than projected image_embeds)
                emb = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(emb)
                del inputs, outputs, emb
        del pil_crops
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
        stride=30 with ~20 players/frame gives ~500 crops for 750 frames.
        """
        all_embeddings = []
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
                all_embeddings.append(self.extract_features(crops))
                sampled += len(crops)

        if not all_embeddings:
            print("⚠ No crops collected for fitting!")
            return

        all_embeddings = np.concatenate(all_embeddings)
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

        del all_embeddings, projections
        gc.collect()

    def classify_from_video(self, tracks, frame_generator, team_colors,
                            sample_stride: int = 10) -> None:
        """
        Assign teams to *all* frames using cached player-ID lookups.

        Only every `sample_stride`-th frame runs model inference.
        In between, teams are propagated from the cache by player-ID.
        """
        id_cache: dict[int, int] = {}   # player_id → team (0 or 1)
        total = len(tracks['players'])

        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= total:
                break

            player_track = tracks['players'][frame_num]
            is_inference_frame = (frame_num % sample_stride == 0)

            if is_inference_frame:
                # ---------- model inference ----------
                crops, xys, pids = [], [], []
                for pid, info in player_track.items():
                    x1, y1, x2, y2 = map(int, info['bbox'])
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crops.append(crop)
                    xys.append([(x1 + x2) / 2, y2])
                    pids.append(pid)

                if crops:
                    emb = self.extract_features(crops)
                    proj = self.reducer.transform(emb)
                    cids = self.cluster_model.predict(proj)
                    teams = self._resolve_gk(cids, np.array(xys))
                    del emb, proj, cids

                    for pid, tid in zip(pids, teams):
                        id_cache[pid] = int(tid)
                        tracks['players'][frame_num][pid]['team'] = int(tid)
                        tracks['players'][frame_num][pid]['team_color'] = \
                            team_colors.get(int(tid), (255, 255, 255))
                else:
                    is_inference_frame = False

            if not is_inference_frame:
                # ---------- cache propagation ----------
                for pid, info in player_track.items():
                    tid = id_cache.get(pid, 0)
                    tracks['players'][frame_num][pid]['team'] = tid
                    tracks['players'][frame_num][pid]['team_color'] = \
                        team_colors.get(tid, (255, 255, 255))

            # Progress
            if (frame_num + 1) % 100 == 0 or (frame_num + 1) == total:
                print(f"  [Team Assignment] Frame {frame_num + 1} / {total}")

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