import os
import gc
from collections import Counter, defaultdict
from typing import Generator, Iterable, List, Literal, TypeVar

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP2_BASE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'model', 'siglip2-base'
)
SIGLIP_SO400M_NAFLEX_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'model', 'siglip-so400m-naflex'
)

HF_MODEL_IDS = {
    "siglip2-base":  "google/siglip2-base-patch16-224",
    "so400m-naflex": "google/siglip-so400m-patch14-naflex",
}

ModelType = Literal["siglip2-base", "so400m-naflex"]

# Optimal batch sizes per device
_DEFAULT_BATCH = {"cuda": 256, "cpu": 16}
# PCA target dim before UMAP — reduces UMAP complexity from O(N*768) to O(N*64)
_PCA_DIM = 64


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


class TeamClassifier2:
    """
    Billing-optimised team classifier backed by SigLIP2.

    Key optimisations vs. the FashionCLIP version
    -----------------------------------------------
    1. torch.compile        — fuses ops, ~30-40% throughput gain on GPU.
    2. torch.inference_mode — cheaper than no_grad; skips autograd entirely.
    3. PCA pre-reduction    — 768d → 64d before UMAP, making fit 5-10x
                              faster and transform nearly instant.
    4. Bulk UMAP transform  — all sampled-frame embeddings collected first,
                              projected in ONE call. UMAP.transform() has
                              high per-call overhead; this eliminates it.
    5. MiniBatchKMeans      — 3-5x faster fit than KMeans on large N.
    6. FP16 by default      — halves GPU memory bandwidth per forward pass.
    7. Auto batch_size      — sensible defaults per device type.
    8. Pinned memory        — zero-copy CPU→GPU pixel_values transfers.
    """

    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = None,
        model_path: str = None,
        use_fp16: bool = True,          # default ON — almost always a win on GPU
        model_type: ModelType = "siglip2-base",
        compile_model: bool = True,     # set False for PyTorch < 2.0
        pca_dim: int = _PCA_DIM,
    ):
        self.device = device
        self.use_fp16 = use_fp16 and ('cuda' in device)
        self.model_type = model_type
        self.pca_dim = pca_dim

        device_key = "cuda" if "cuda" in device else "cpu"
        self.batch_size = batch_size if batch_size else _DEFAULT_BATCH[device_key]

        # ----------------------------------------------------------------
        # Resolve model path: local checkpoint → HF hub fallback
        # ----------------------------------------------------------------
        if model_path:
            resolved_path = model_path
        elif model_type == "siglip2-base":
            resolved_path = (
                SIGLIP2_BASE_MODEL_PATH
                if os.path.isdir(SIGLIP2_BASE_MODEL_PATH)
                else HF_MODEL_IDS["siglip2-base"]
            )
        else:
            resolved_path = (
                SIGLIP_SO400M_NAFLEX_MODEL_PATH
                if os.path.isdir(SIGLIP_SO400M_NAFLEX_MODEL_PATH)
                else HF_MODEL_IDS["so400m-naflex"]
            )

        print(f"  [TeamClassifier] Loading {model_type} | device={device} | "
              f"fp16={self.use_fp16} | compile={compile_model} | "
              f"batch={self.batch_size}")

        # ----------------------------------------------------------------
        # Model — SigLIP vision encoder
        # ----------------------------------------------------------------
        model = SiglipVisionModel.from_pretrained(resolved_path)
        if self.use_fp16:
            model = model.half()
        model = model.to(device).eval()

        # torch.compile: fuses kernels, reduces Python dispatch overhead.
        # reduce-overhead: best for fixed-size inputs (siglip2-base @ 224px).
        # default: safer for NaFlex variable-shape inputs to avoid recompiles.
        if compile_model and hasattr(torch, 'compile'):
            compile_mode = (
                "reduce-overhead" if model_type == "siglip2-base" else "default"
            )
            self.features_model = torch.compile(model, mode=compile_mode)
            print(f"    torch.compile mode='{compile_mode}' applied.")
        else:
            self.features_model = model

        self.processor = AutoProcessor.from_pretrained(resolved_path)

        # ----------------------------------------------------------------
        # Clustering pipeline: embeds → PCA(64d) → UMAP(3d) → KMeans
        # ----------------------------------------------------------------
        self.pca = PCA(n_components=pca_dim, whiten=True)
        self.reducer = umap.UMAP(
            n_components=3,
            n_neighbors=30,    # larger = more global structure preserved
            min_dist=0.1,
            metric='cosine',   # correct for L2-normalised embeddings
            random_state=42,
            low_memory=True,   # reduces peak RAM during fit
        )
        self.cluster_model = MiniBatchKMeans(
            n_clusters=5, random_state=42, n_init=10, batch_size=1024
        )

        self.team_0_cluster = None
        self.team_1_cluster = None
        self.gk_clusters = []
        self.team_fingerprints = {}
        self.cluster_to_team = {} # map: cluster_id -> team_id (0, 1, or -1 for GK)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
        h, w, _ = crop.shape
        if h < 10 or w < 10:
            return crop
        return crop[int(h * 0.2):int(h * 0.5), int(w * 0.2):int(w * 0.8), :]

    @staticmethod
    def _get_color_stats(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.array([0, 0, 0])
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return np.mean(hsv, axis=(0, 1))

    def _to_pil(self, crop: np.ndarray):
        # siglip2-base: fixed 224px input, resize early to save memory.
        # so400m-naflex: keep native size — processor handles flexible grids.
        if self.model_type == "siglip2-base":
            crop = cv2.resize(crop, (224, 224))
        return sv.cv2_to_pillow(crop)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Encode crops → L2-normalised (N, D) float32 embeddings.

        Uses pooler_output (mean-pool + LayerNorm) — SigLIP's canonical
        representation, better than raw last_hidden_state mean-pooling.
        Pinned memory used on CUDA for zero-copy H2D transfers.
        """
        pil_crops = [
            self._to_pil(self._preprocess_crop(c))
            for c in crops if c.size > 0
        ]

        data = []
        use_cuda = 'cuda' in self.device

        # torch.inference_mode: skips autograd graph construction entirely —
        # cheaper than no_grad(), which still tracks some internal state.
        with torch.inference_mode():
            for batch in create_batches(pil_crops, self.batch_size):
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                    padding="max_length",   # required by SigLIP processors
                )
                pixel_values = inputs['pixel_values']

                if use_cuda:
                    # pin_memory + non_blocking: overlaps H2D copy with CPU
                    # preprocessing of the next batch — hides transfer latency.
                    pixel_values = pixel_values.pin_memory().to(
                        self.device, non_blocking=True
                    )
                else:
                    pixel_values = pixel_values.to(self.device)

                if self.use_fp16:
                    pixel_values = pixel_values.half()

                outputs = self.features_model(pixel_values=pixel_values)
                emb = outputs.pooler_output.cpu().float().numpy()

                # L2-normalise so cosine distance == euclidean distance.
                # Required because UMAP metric='cosine' and KMeans both
                # implicitly assume euclidean — normalising makes them consistent.
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                data.append(emb / np.clip(norms, 1e-8, None))

                del pixel_values, outputs, emb

        del pil_crops
        gc.collect()
        return np.concatenate(data)

    def _embed_and_project(self, embeddings: np.ndarray) -> np.ndarray:
        """PCA → UMAP transform. Both are instant post-fit."""
        return self.reducer.transform(self.pca.transform(embeddings))

    def _resolve_gk(
        self, cluster_ids: np.ndarray, players_xy: np.ndarray
    ) -> np.ndarray:
        team_0_mask = cluster_ids == self.team_0_cluster
        team_1_mask = cluster_ids == self.team_1_cluster
        t0_cent = (players_xy[team_0_mask].mean(axis=0)
                   if np.any(team_0_mask) else np.zeros(2))
        t1_cent = (players_xy[team_1_mask].mean(axis=0)
                   if np.any(team_1_mask) else np.zeros(2))
        out = np.empty(len(cluster_ids), dtype=int)
        for i, cid in enumerate(cluster_ids):
            if cid == self.team_0_cluster:
                out[i] = 0
            elif cid == self.team_1_cluster:
                out[i] = 1
            else:
                out[i] = 0 if (np.linalg.norm(players_xy[i] - t0_cent)
                               < np.linalg.norm(players_xy[i] - t1_cent)) else 1
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_from_video(
        self, tracks_players, frame_generator, sample_stride: int = 30
    ) -> None:
        """
        Fit PCA → UMAP → KMeans on sampled frames.

        Cost tip: stride=30 on 30fps = 1 reference frame/sec.
        Raise to 60-90 on long clips to halve fit time with minimal quality loss.
        """
        all_embeddings = []
        all_hsv_stats = []

        for frame_num, frame in enumerate(frame_generator):
            if frame_num % sample_stride != 0:
                continue
            if frame_num >= len(tracks_players):
                break

            crops = []
            h_f, w_f = frame.shape[:2]
            for track in tracks_players[frame_num].values():
                x1, y1, x2, y2 = map(int, track['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1: continue
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

            if crops:
                processed = [self._preprocess_crop(c) for c in crops]
                all_hsv_stats.append(
                    np.array([self._get_color_stats(p) for p in processed])
                )
                all_embeddings.append(self.extract_features(crops))

        if not all_embeddings:
            print("⚠ No crops collected for fitting!")
            return

        all_embeddings = np.concatenate(all_embeddings)
        all_hsv_stats = np.concatenate(all_hsv_stats)
        N, D = all_embeddings.shape
        print(f"  Fitting pipeline on {N} crops (stride={sample_stride})...")

        # PCA: 768d → 64d. UMAP complexity is O(N*D) — this is a 12x speedup
        # on UMAP fit and makes every subsequent .transform() call near-instant.
        print(f"    [1/3] PCA {D}d → {self.pca_dim}d ...")
        reduced = self.pca.fit_transform(all_embeddings)

        print(f"    [2/3] UMAP {self.pca_dim}d → 3d ...")
        projections = self.reducer.fit_transform(reduced)

        print(f"    [3/3] MiniBatchKMeans ...")
        self.cluster_model.fit(projections)

        labels = self.cluster_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]

        self.team_0_cluster = sorted_clusters[0]
        self.team_1_cluster = sorted_clusters[1]
        self.gk_clusters = sorted_clusters[2:].tolist()

        self.team_fingerprints = {}
        def color_dist(hsv1, hsv2):
            dh = min(abs(hsv1[0]-hsv2[0]), 180-abs(hsv1[0]-hsv2[0])) / 180.0
            ds, dv = abs(hsv1[1]-hsv2[1]) / 255.0, abs(hsv1[2]-hsv2[2]) / 255.0
            # Weight Hue significantly more (25.0 weight on squared diff)
            return np.sqrt(25.0*dh*dh + ds*ds + dv*dv)

        for cid in unique:
            mask = labels == cid
            self.team_fingerprints[int(cid)] = np.mean(all_hsv_stats[mask], axis=0)

        # ----------------------------------------------------------------
        # 4. Advanced Anchor Selection (Diversity + Determinism)
        # ----------------------------------------------------------------
        anc0_idx = sorted_clusters[0]
        anc0_hsv = self.team_fingerprints[anc0_idx]
        
        # Find Anchor 1: must be visually distinct from Anchor 0
        best_anc1_idx = sorted_clusters[1]
        max_d = -1
        for cid in sorted_clusters[1:]:
             if counts[unique == cid][0] < (N * 0.10): continue 
             d = color_dist(anc0_hsv, self.team_fingerprints[cid])
             if d > max_d:
                 max_d = d
                 best_anc1_idx = cid
        
        # Deterministic sorting: map lower hue (Green) to Team 0 (Blue) consistently
        h0, h1 = anc0_hsv[0], self.team_fingerprints[best_anc1_idx][0]
        if h0 < h1:
            self.team_0_cluster, self.team_1_cluster = anc0_idx, best_anc1_idx
        else:
            self.team_0_cluster, self.team_1_cluster = best_anc1_idx, anc0_idx

        # ----------------------------------------------------------------
        # 5. Hybrid Mapping Logic
        # ----------------------------------------------------------------
        anchor0_hsv = self.team_fingerprints[self.team_0_cluster]
        anchor1_hsv = self.team_fingerprints[self.team_1_cluster]
        self.gk_clusters = []
        self.cluster_to_team = {}
        
        print("    Cluster Mapping (Spectral Robustness):")
        for cid in unique:
            hsv = self.team_fingerprints[cid]
            d0, d1 = color_dist(hsv, anchor0_hsv), color_dist(hsv, anchor1_hsv)
            n = counts[unique == cid][0]
            
            # If cluster is far from both teams, mark as GK (handled by position)
            # Threshold 0.8 is strict enough to catch GKs (Orange/Neon)
            if d0 > 0.8 and d1 > 0.8:
                self.cluster_to_team[int(cid)] = -1 # Special GK status
                self.gk_clusters.append(int(cid))
                tag = "GK/Mystery"
            else:
                tid = 0 if d0 < d1 else 1
                self.cluster_to_team[int(cid)] = tid
                tag = f"ANCHOR {tid}" if cid in [self.team_0_cluster, self.team_1_cluster] else f"Team {tid}"
            
            print(f"    - Cluster {cid} ({tag}, n={n}): H={hsv[0]:.1f} S={hsv[1]:.1f} V={hsv[2]:.1f}")

        del all_embeddings, reduced, projections, all_hsv_stats
        gc.collect()

    def classify_from_video(
        self, tracks, frame_generator, team_colors, sample_stride: int = 5
    ) -> None:
        """
        Assign teams using Adaptive Fingerprinting Resolution.

        Critical optimisation: all embeddings are buffered across ALL sampled
        frames, then projected via PCA + UMAP in a SINGLE batch call.
        Old approach: UMAP.transform() called ~(total/stride) times.
        New approach: called exactly once. High per-call overhead eliminated.
        """
        total = len(tracks['players'])

        # ----------------------------------------------------------------
        # Pass 1: collect all embeddings in one sweep
        # ----------------------------------------------------------------
        print(f"  [Pass 1] Extracting embeddings (stride={sample_stride})...")

        frame_records = []   # (pid, xy, hsv, embedding_index)
        all_embeddings = []
        emb_idx = 0

        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= total:
                break
            if frame_num % sample_stride != 0:
                continue

            crops, xys, pids = [], [], []
            h_f, w_f = frame.shape[:2]
            for pid, info in tracks['players'][frame_num].items():
                x1, y1, x2, y2 = map(int, info['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1: continue

                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size == 0:
                    continue
                crops.append(raw_crop)
                xys.append([(x1 + x2) / 2, y2])
                pids.append(pid)

            if not crops:
                continue

            torso_crops = [self._preprocess_crop(c) for c in crops]
            hsv_stats   = [self._get_color_stats(tc) for tc in torso_crops]
            embs        = self.extract_features(crops)

            for pid, xy, hsv in zip(pids, xys, hsv_stats):
                frame_records.append((pid, xy, hsv, emb_idx))
                emb_idx += 1

            all_embeddings.append(embs)

            if (frame_num + 1) % 200 == 0:
                print(f"    Frame {frame_num + 1}/{total} | crops: {emb_idx}")

        if not all_embeddings:
            print("⚠ No crops collected for classification!")
            return

        # ----------------------------------------------------------------
        # Single bulk PCA + UMAP transform — the core billing optimisation
        # ----------------------------------------------------------------
        print(f"  [Pass 1] Bulk projecting {emb_idx} embeddings (PCA→UMAP)...")
        emb_matrix = np.concatenate(all_embeddings)
        del all_embeddings
        gc.collect()

        all_cids = self.cluster_model.predict(
            self.reducer.transform(
                self.pca.transform(emb_matrix)
            )
        )
        del emb_matrix

        # ----------------------------------------------------------------
        # Pass 2: aggregate evidence per track ID
        # ----------------------------------------------------------------
        print("  [Pass 2] Aggregating track evidence...")
        track_evidence = defaultdict(list)
        for pid, xy, hsv, idx in frame_records:
            track_evidence[pid].append({
                "cid": int(all_cids[idx]),
                "xy":  xy,
                "hsv": hsv,
            })
        del all_cids, frame_records

        t0_prof = self.team_fingerprints.get(self.team_0_cluster, np.zeros(3))
        t1_prof = self.team_fingerprints.get(self.team_1_cluster, np.zeros(3))

        all_t0_xys = [ev['xy'] for evs in track_evidence.values()
                      for ev in evs if ev['cid'] == self.team_0_cluster]
        all_t1_xys = [ev['xy'] for evs in track_evidence.values()
                      for ev in evs if ev['cid'] == self.team_1_cluster]
        t0_cent = np.mean(all_t0_xys, axis=0) if all_t0_xys else np.zeros(2)
        t1_cent = np.mean(all_t1_xys, axis=0) if all_t1_xys else np.zeros(2)

        def color_dist(h1, h2):
            dh = min(abs(h1[0]-h2[0]), 180-abs(h1[0]-h2[0])) / 180.0
            ds, dv = abs(h1[1]-h2[1]) / 255.0, abs(h1[2]-h2[2]) / 255.0
            return np.sqrt(25.0*dh*dh + ds*ds + dv*dv)

        anchor0_hsv = self.team_fingerprints.get(self.team_0_cluster, np.zeros(3))
        anchor1_hsv = self.team_fingerprints.get(self.team_1_cluster, np.zeros(3))

        final_assignments = {}
        for pid, evidence in track_evidence.items():
            avg_hsv  = np.mean([ev['hsv'] for ev in evidence], axis=0)
            votes    = [self.cluster_to_team.get(ev['cid'], 0) for ev in evidence]
            majority_tid = Counter(votes).most_common(1)[0][0]
            
            # --- Resolution Engine ---
            # 1. If majority is a clear Outfield Team (0 or 1), trust the mapping.
            #    (Added a stricter guardrail to flip if color is decisively different)
            d0, d1 = color_dist(avg_hsv, anchor0_hsv), color_dist(avg_hsv, anchor1_hsv)
            
            if majority_tid != -1:
                # Spectral Shield: overrule only if color is extremely convincing (~2.5x difference)
                if d1 < d0 * 0.4: tid = 1
                elif d0 < d1 * 0.4: tid = 0
                else: tid = majority_tid
            else:
                # 2. If majority is a GK/Mystery cluster, use Positional Centroids
                avg_xy = np.mean([ev['xy'] for ev in evidence], axis=0)
                tid = 0 if np.linalg.norm(avg_xy - t0_cent) < np.linalg.norm(avg_xy - t1_cent) else 1
            
            final_assignments[pid] = tid

        # ----------------------------------------------------------------
        # Pass 3: write to all frames
        # ----------------------------------------------------------------
        print("  [Pass 3] Writing assignments...")
        for frame_num in range(total):
            for pid, info in tracks['players'][frame_num].items():
                tid = final_assignments.get(pid, 0)
                tracks['players'][frame_num][pid]['team'] = tid
                tracks['players'][frame_num][pid]['team_color'] = \
                    team_colors.get(tid, (255, 255, 255))

        print(f"  Done. Resolved {len(final_assignments)} unique players.")

    def release_model(self):
        del self.features_model
        del self.processor
        gc.collect()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        print(f"  [Memory] Released {self.model_type} + CUDA cache cleared.")

    # ------------------------------------------------------------------
    # Legacy wrappers
    # ------------------------------------------------------------------
    def fit(self, crops: List[np.ndarray]) -> None:
        embs    = self.extract_features(crops)
        reduced = self.pca.fit_transform(embs)
        proj    = self.reducer.fit_transform(reduced)
        self.cluster_model.fit(proj)
        labels  = self.cluster_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]
        self.team_0_cluster = sorted_clusters[0]
        self.team_1_cluster = sorted_clusters[1]
        self.gk_clusters    = sorted_clusters[2:].tolist()

    def predict(self, crops, players_xy):
        return self.predict_frame(crops, players_xy)

    def predict_frame(self, frame_crops, players_xy):
        if len(frame_crops) == 0:
            return np.array([])
        embs = self.extract_features(frame_crops)
        proj = self._embed_and_project(embs)
        cids = self.cluster_model.predict(proj)
        return self._resolve_gk(cids, players_xy)