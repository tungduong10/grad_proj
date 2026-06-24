import numpy as np
from typing import List
from collections import defaultdict
from .base_team_assigner import BaseTeamClassifier

class FootballTeamClassifier(BaseTeamClassifier):
    """
    Football-specific team classifier.
    - Crops only the upper torso (height 0.2 to 0.5) to avoid grass and shorts.
    - Expects 5 clusters (2 teams, 2 GKs, 1 referee/noise).
    - Uses spatial gravity to map goalkeepers back to their defending team.
    """
    
    def _get_expected_clusters(self) -> int:
        return 5

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        h, w, _ = crop.shape
        if h < 10 or w < 10:
            return crop
        return crop[int(h * 0.2):int(h * 0.5), int(w * 0.2):int(w * 0.8), :]

    def _map_clusters_to_teams(self, unique, counts, N):
        anc0_idx = unique[0]
        anc0_hsv = self.team_fingerprints[anc0_idx]
        
        # Find Anchor 1: must be visually distinct from Anchor 0
        best_anc1_idx = unique[1]
        max_d = -1
        for cid in unique[1:]:
             if counts[unique == cid][0] < (N * 0.10): continue 
             d = self._color_dist(anc0_hsv, self.team_fingerprints[cid])
             if d > max_d:
                 max_d = d
                 best_anc1_idx = cid
        
        # Deterministic sorting: map lower hue to Team 0 consistently
        h0, h1 = anc0_hsv[0], self.team_fingerprints[best_anc1_idx][0]
        if h0 < h1:
            self.team_0_cluster, self.team_1_cluster = anc0_idx, best_anc1_idx
        else:
            self.team_0_cluster, self.team_1_cluster = best_anc1_idx, anc0_idx

        anchor0_hsv = self.team_fingerprints[self.team_0_cluster]
        anchor1_hsv = self.team_fingerprints[self.team_1_cluster]
        
        self.gk_clusters = []
        self.cluster_to_team = {}
        
        print("    Cluster Mapping (Spectral Robustness):")
        for cid in unique:
            hsv = self.team_fingerprints[cid]
            d0, d1 = self._color_dist(hsv, anchor0_hsv), self._color_dist(hsv, anchor1_hsv)
            n = counts[unique == cid][0]
            
            # If cluster is far from both teams, mark as GK/Mystery (Threshold 0.8)
            if d0 > 0.8 and d1 > 0.8:
                self.cluster_to_team[int(cid)] = -1
                self.gk_clusters.append(int(cid))
                tag = "GK/Mystery"
            else:
                tid = 0 if d0 < d1 else 1
                self.cluster_to_team[int(cid)] = tid
                tag = f"ANCHOR {tid}" if cid in [self.team_0_cluster, self.team_1_cluster] else f"Team {tid}"
            
            print(f"    - Cluster {cid} ({tag}, n={n}): H={hsv[0]:.1f} S={hsv[1]:.1f} V={hsv[2]:.1f}")

    def _get_raw_tids(self, evidence, frame_t0_xys, frame_t1_xys):
        # 1. Dynamic GK resolution using spatial gravity (proximity + density)
        t0_gravity = 0.0
        t1_gravity = 0.0
        
        for ev in evidence:
            f = ev['frame']
            gk_xy = np.array(ev['xy'])
            
            for xy in frame_t0_xys.get(f, []):
                dist = np.linalg.norm(xy - gk_xy)
                t0_gravity += 1.0 / (dist + 50.0)
                
            for xy in frame_t1_xys.get(f, []):
                dist = np.linalg.norm(xy - gk_xy)
                t1_gravity += 1.0 / (dist + 50.0)
                
        gk_tid = 0 if t0_gravity >= t1_gravity else 1

        # 2. Extract RAW tids sequentially
        raw_tids = {}
        anchor0_hsv = self.team_fingerprints.get(self.team_0_cluster, np.zeros(3))
        anchor1_hsv = self.team_fingerprints.get(self.team_1_cluster, np.zeros(3))

        for ev in evidence:
            cid = ev['cid']
            base_tid = self.cluster_to_team.get(cid, 0)
            
            # Spectral override for tricky lighting
            d0 = self._color_dist(ev['hsv'], anchor0_hsv)
            d1 = self._color_dist(ev['hsv'], anchor1_hsv)
            
            if base_tid != -1:
                if d1 < d0 * 0.4:   tid = 1
                elif d0 < d1 * 0.4: tid = 0
                else:               tid = base_tid
            else:
                tid = gk_tid

            raw_tids[ev['frame']] = tid

        return raw_tids
