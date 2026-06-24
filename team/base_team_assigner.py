import os
import gc
from collections import Counter, defaultdict
from typing import Generator, Iterable, List, Literal, TypeVar
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from transformers import AutoProcessor, SiglipVisionModel, SiglipModel

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

_DEFAULT_BATCH = {"cuda": 64, "cpu": 16}
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

class BaseTeamClassifier:
    """
    Base object-oriented team classifier backed by SigLIP2.
    Sport-specific subclasses should implement _preprocess_crop, 
    _get_expected_clusters, _map_clusters_to_teams, and _get_raw_tids.
    """

    def __init__(
        self,
        device: str = 'cpu',
        batch_size: int = None,
        model_path: str = None,
        use_fp16: bool = True,
        model_type: ModelType = "siglip2-base",
        compile_model: bool = True,
        pca_dim: int = _PCA_DIM,
        reid_max_dist: float = 80,
        reid_max_frames: int = 30,
        use_text_encoder: bool = False,
    ):
        self.device = device
        self.use_fp16 = use_fp16 and ('cuda' in device)
        self.model_type = model_type
        self.pca_dim = pca_dim
        self.use_text_encoder = use_text_encoder

        device_key = "cuda" if "cuda" in device else "cpu"
        self.batch_size = batch_size if batch_size else _DEFAULT_BATCH[device_key]

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

        mode_str = "text+vision" if use_text_encoder else "vision-only"
        print(f"  [{self.__class__.__name__}] Loading {model_type} ({mode_str}) | "
              f"device={device} | fp16={self.use_fp16} | compile={compile_model} | "
              f"batch={self.batch_size}")

        if use_text_encoder:
            full_model = SiglipModel.from_pretrained(resolved_path)
            if self.use_fp16:
                full_model = full_model.half()
            full_model = full_model.to(device).eval()
            self.full_model = full_model
            vision_model = full_model.vision_model
        else:
            self.full_model = None
            vision_model = SiglipVisionModel.from_pretrained(resolved_path)
            if self.use_fp16:
                vision_model = vision_model.half()
            vision_model = vision_model.to(device).eval()

        if compile_model and hasattr(torch, 'compile'):
            compile_mode = (
                "reduce-overhead" if model_type == "siglip2-base" else "default"
            )
            self.features_model = torch.compile(vision_model, mode=compile_mode)
            print(f"    torch.compile mode='{compile_mode}' applied.")
        else:
            self.features_model = vision_model

        self.processor = AutoProcessor.from_pretrained(resolved_path)

        # Text-encoder mode skips PCA/UMAP/KMeans (not needed for zero-shot)
        if not use_text_encoder:
            self.pca = PCA(n_components=pca_dim, whiten=True)
            self.reducer = umap.UMAP(
                n_components=3,
                n_neighbors=30,
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                low_memory=True,  
            )
            self.cluster_model = MiniBatchKMeans(
                n_clusters=self._get_expected_clusters(), random_state=42, n_init=10, batch_size=1024
            )

        self.team_0_cluster = None
        self.team_1_cluster = None
        self.gk_clusters = []
        self.team_fingerprints = {}
        self.cluster_to_team = {}
        self._text_embeddings = None

    # --- Methods to be overridden by subclasses ---
    def _get_expected_clusters(self) -> int:
        raise NotImplementedError

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _map_clusters_to_teams(self, unique, counts, N):
        raise NotImplementedError

    def _get_raw_tids(self, evidence, frame_t0_xys, frame_t1_xys):
        raise NotImplementedError
    # ----------------------------------------------

    @staticmethod
    def _bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(float)

    @staticmethod
    def _get_bgr_stats(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.array([0, 0, 0])
        return np.mean(crop, axis=(0, 1))

    @staticmethod
    def _isheavilyoccluded(target_bbox: list, all_bboxes: list, threshold: float=0.35) -> bool:
        tx1, ty1, tx2, ty2 = target_bbox
        target_area = max(0, tx2-tx1)*max(0, ty2-ty1)
        if target_area == 0:
            return True

        for bbox in all_bboxes:
            if np.array_equal(bbox, target_bbox):
                continue
            x1, y1, x2, y2 = bbox
            ix1, iy1 = max(tx1, x1), max(ty1, y1)
            ix2, iy2 = min(tx2, x2), min(ty2, y2)
            i_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if (i_area / target_area) > threshold:
                return True
        return False

    @staticmethod
    def _evidence_weight(hsv: np.ndarray) -> float:
        s_norm = hsv[1] / 255.0
        v_norm = hsv[2] / 255.0
        return float(s_norm * v_norm)
        
    def _to_pil(self, crop: np.ndarray):
        if self.model_type == "siglip2-base":
            crop = cv2.resize(crop, (224, 224))
        return sv.cv2_to_pillow(crop)

    # ------ Zero-shot text-prompt helpers ------

    def precompute_text_embeddings(self, text_labels: List[str]):
        """Pre-compute and cache normalized text embeddings for zero-shot."""
        if self.full_model is None:
            raise RuntimeError("Text encoder not loaded. Set use_text_encoder=True.")
        text_inputs = self.processor(
            text=text_labels, padding="max_length", return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        with torch.inference_mode():
            text_features = self.full_model.get_text_features(**text_inputs)
            if hasattr(text_features, 'pooler_output'):
                text_features = text_features.pooler_output
            elif hasattr(text_features, 'text_embeds'):
                text_features = text_features.text_embeds
                
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            if self.use_fp16:
                text_features = text_features.half()
        self._text_embeddings = text_features
        print(f"    Pre-computed text embeddings for {len(text_labels)} labels.")

    def zero_shot_classify(self, crops: List[np.ndarray]) -> np.ndarray:
        """Classify a list of BGR crops against pre-computed text embeddings.
        Returns array of label indices (0, 1, ...) for each crop."""
        if self._text_embeddings is None:
            raise RuntimeError("Call precompute_text_embeddings() first.")
        pil_crops = [self._to_pil(c) for c in crops if c.size > 0]
        if not pil_crops:
            return np.array([], dtype=int)

        all_preds = []
        use_cuda = 'cuda' in self.device
        with torch.inference_mode():
            for batch in create_batches(pil_crops, self.batch_size):
                inputs = self.processor(
                    images=batch, return_tensors="pt", padding="max_length"
                )
                pixel_values = inputs['pixel_values']
                if use_cuda:
                    pixel_values = pixel_values.pin_memory().to(
                        self.device, non_blocking=True
                    )
                else:
                    pixel_values = pixel_values.to(self.device)
                if self.use_fp16:
                    pixel_values = pixel_values.half()

                image_features = self.full_model.get_image_features(pixel_values=pixel_values)
                if hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
                elif hasattr(image_features, 'image_embeds'):
                    image_features = image_features.image_embeds
                    
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Cosine similarity → pick highest
                similarity = image_features @ self._text_embeddings.T
                preds = similarity.argmax(dim=-1).cpu().numpy()
                all_preds.append(preds)
                del pixel_values, image_features, similarity

        return np.concatenate(all_preds)

    @staticmethod
    def _color_dist(hsv1, hsv2):
        dh = min(abs(hsv1[0]-hsv2[0]), 180-abs(hsv1[0]-hsv2[0])) / 180.0
        ds, dv = abs(hsv1[1]-hsv2[1]) / 255.0, abs(hsv1[2]-hsv2[2]) / 255.0
        return np.sqrt(25.0*dh*dh + ds*ds + dv*dv)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        pil_crops = [
            self._to_pil(self._preprocess_crop(c))
            for c in crops if c.size > 0
        ]
        data = []
        use_cuda = 'cuda' in self.device

        with torch.inference_mode():
            for batch in create_batches(pil_crops, self.batch_size):
                inputs = self.processor(
                    images=batch,
                    return_tensors="pt",
                    padding="max_length",
                )
                pixel_values = inputs['pixel_values']

                if use_cuda:
                    pixel_values = pixel_values.pin_memory().to(
                        self.device, non_blocking=True
                    )
                else:
                    pixel_values = pixel_values.to(self.device)

                if self.use_fp16:
                    pixel_values = pixel_values.half()

                outputs = self.features_model(pixel_values=pixel_values)
                emb = outputs.pooler_output.cpu().float().numpy()
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                data.append(emb / np.clip(norms, 1e-8, None))
                del pixel_values, outputs, emb

        del pil_crops
        gc.collect()
        return np.concatenate(data)

    def _embed_and_project(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.transform(self.pca.transform(embeddings))

    def _apply_assignments(self, tracks, final_assignments, team_colors):
        total = len(tracks['players'])
        for frame_num in range(total):
            for pid in tracks['players'][frame_num].keys():
                tid = final_assignments.get(pid, {}).get(frame_num, 0)
                tracks['players'][frame_num][pid]['team'] = tid
                tracks['players'][frame_num][pid]['team_color'] = \
                    team_colors.get(tid, (255, 255, 255))

    def fit_from_video(
        self, tracks_players, frame_generator, sample_stride: int = 30
    ) -> None:
        all_embeddings = []
        all_hsv_stats = []

        for frame_num, frame in enumerate(frame_generator):
            if frame_num % sample_stride != 0:
                continue
            if frame_num >= len(tracks_players):
                break

            crops = []
            h_f, w_f = frame.shape[:2]
            curr_boxes = [track['bbox'] for track in tracks_players[frame_num].values()]
            for track in tracks_players[frame_num].values():
                if self._isheavilyoccluded(track['bbox'], curr_boxes, threshold=0.15):
                    continue

                x1, y1, x2, y2 = map(int, track['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1: continue
                
                crop_area = (x2-x1)*(y2-y1)
                if crop_area < 400: continue

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)

            if crops:
                processed = [self._preprocess_crop(c) for c in crops]
                all_hsv_stats.append(
                    np.array([self._get_bgr_stats(p) for p in processed])
                )
                all_embeddings.append(self.extract_features(crops))

        if not all_embeddings:
            print("⚠ No crops collected for fitting!")
            return

        all_embeddings = np.concatenate(all_embeddings)
        all_hsv_stats = np.concatenate(all_hsv_stats)
        N, D = all_embeddings.shape
        print(f"  Fitting pipeline on {N} crops (stride={sample_stride})...")

        print(f"    [1/3] PCA {D}d → {self.pca_dim}d ...")
        reduced = self.pca.fit_transform(all_embeddings)

        print(f"    [2/3] UMAP {self.pca_dim}d → 3d ...")
        projections = self.reducer.fit_transform(reduced)

        print(f"    [3/3] MiniBatchKMeans ...")
        self.cluster_model.fit(projections)

        labels = self.cluster_model.labels_
        unique, counts = np.unique(labels, return_counts=True)
        sorted_clusters = unique[np.argsort(-counts)]

        self.team_fingerprints = {}
        for cid in unique:
            mask = labels == cid
            cluster_mean_bgr = np.mean(all_hsv_stats[mask], axis=0)
            self.team_fingerprints[int(cid)] = self._bgr_to_hsv(cluster_mean_bgr)

        self._map_clusters_to_teams(sorted_clusters, counts, N)

        del all_embeddings, reduced, projections, all_hsv_stats
        gc.collect()

    def classify_from_video(
        self, tracks, frame_generator, team_colors, sample_stride: int = 5
    ) -> None:
        total = len(tracks['players'])
        print(f"  [Pass 1] Extracting embeddings (stride={sample_stride})...")

        frame_records = []
        all_embeddings = []
        emb_idx = 0

        for frame_num, frame in enumerate(frame_generator):
            if frame_num >= total:
                break
            if frame_num % sample_stride != 0:
                continue

            crops, xys, pids = [], [], []
            h_f, w_f = frame.shape[:2]
            curr_boxes = [track['bbox'] for track in tracks['players'][frame_num].values()]

            for pid, info in tracks['players'][frame_num].items():
                if self._isheavilyoccluded(info['bbox'], curr_boxes, threshold=0.20):
                    continue
                x1, y1, x2, y2 = map(int, info['bbox'])
                x1, y1 = max(0, min(x1, w_f)), max(0, min(y1, h_f))
                x2, y2 = max(0, min(x2, w_f)), max(0, min(y2, h_f))
                if x2 <= x1 or y2 <= y1: continue

                crop_area = (x2-x1)*(y2-y1)
                if crop_area < 400: continue

                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop.size == 0:
                    continue
                crops.append(raw_crop)
                xys.append([(x1 + x2) / 2, y2])
                pids.append(pid)

            if not crops:
                continue

            torso_crops = [self._preprocess_crop(c) for c in crops]
            hsv_stats   = [self._bgr_to_hsv(self._get_bgr_stats(tc)) for tc in torso_crops]
            embs        = self.extract_features(crops)

            for pid, xy, hsv in zip(pids, xys, hsv_stats):
                frame_records.append((pid, xy, hsv, emb_idx, frame_num))
                emb_idx += 1

            all_embeddings.append(embs)

            if (frame_num + 1) % 200 == 0:
                print(f"    Frame {frame_num + 1}/{total} | crops: {emb_idx}")

        if not all_embeddings:
            print("⚠ No crops collected for classification!")
            return

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

        print("  [Pass 2] Aggregating track evidence...")
        track_evidence = defaultdict(list)
        for pid, xy, hsv, idx, frame_num in frame_records:
            track_evidence[pid].append({
                "cid": int(all_cids[idx]),
                "xy":  xy,
                "hsv": hsv,
                "frame": frame_num,
            })
        del all_cids, frame_records

        frame_t0_xys = defaultdict(list)
        frame_t1_xys = defaultdict(list)
        for evs in track_evidence.values():
            for ev in evs:
                cid = ev['cid']
                tid = self.cluster_to_team.get(cid, -1)
                if tid == 0:
                    frame_t0_xys[ev['frame']].append(np.array(ev['xy']))
                elif tid == 1:
                    frame_t1_xys[ev['frame']].append(np.array(ev['xy']))

        final_assignments = {}
        for pid, evidence in track_evidence.items():
            evidence.sort(key=lambda x: x["frame"])
            
            raw_tids = self._get_raw_tids(evidence, frame_t0_xys, frame_t1_xys)

            known_frames = sorted(raw_tids.keys())
            if not known_frames:
                continue

            filtered_tids = {}
            tids_array = [raw_tids[f] for f in known_frames]
            
            for i, f in enumerate(known_frames):
                start_i = max(0, i - 4)
                end_i = min(len(tids_array), i + 5)
                w = tids_array[start_i:end_i]
                filtered_tids[f] = 0 if w.count(0) > w.count(1) else 1

            final_assignments[pid] = {}
            first_f, last_f = known_frames[0], known_frames[-1]
            
            for f in range(0, first_f):
                final_assignments[pid][f] = filtered_tids[first_f]
            for f in range(last_f + 1, total):
                final_assignments[pid][f] = filtered_tids[last_f]
                
            current_tid = filtered_tids[first_f]
            for f in range(first_f, last_f + 1):
                if f in filtered_tids:
                    current_tid = filtered_tids[f]
                final_assignments[pid][f] = current_tid

        print("  [Pass 3] Writing assignments...")
        self._apply_assignments(tracks, final_assignments, team_colors)
        print(f"  Done. Resolved {len(final_assignments)} unique players.")

    def release_model(self):
        del self.features_model
        del self.processor
        gc.collect()
        if 'cuda' in self.device:
            torch.cuda.empty_cache()
        print(f"  [Memory] Released {self.model_type} + CUDA cache cleared.")