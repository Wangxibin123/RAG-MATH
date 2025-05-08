import numpy as np
from typing import List, Tuple
import faiss # Ensure faiss-cpu or faiss-gpu is installed
import os
from .base import BaseStore
from ..cfg import CFG, ROOT # Import ROOT
from pathlib import Path


class FaissStore(BaseStore):
    def __init__(self,
                 index_path_override: str | None = None,
                 dimension_override: int | None = None):
        """
        index_path_override —— 子类若想用自己的文件路径，在这里传入
        dimension_override  —— 子类若想用自己的向量维度，在这里传入
        """
        base_cfg = CFG.store                  # conf/faiss.yaml

        # ▸ 1. 选路径
        if index_path_override:
            _index_path_str = index_path_override
        else:
            _index_path_str = base_cfg.get("index_path", "models/faiss_index.bin")

        
        # Ensure index_path is absolute or relative to project ROOT
        if not os.path.isabs(_index_path_str):
            self.index_file_path = ROOT / _index_path_str
        else:
            self.index_file_path = Path(_index_path_str) # Convert to Path object

        self.id_map_file_path = self.index_file_path.with_suffix(self.index_file_path.suffix + ".map")
        

        if dimension_override:
            self.dimension = dimension_override
        else:
            self.dimension = CFG.embed_dim

        self.index = None
        self.faiss_ids_map: List[str] = []
        
        print(f"FaissStore initialized. Index path: {self.index_file_path}, Map path: {self.id_map_file_path}")
        self._load_index_and_map() # Attempt to load on initialization

    def _ensure_dir_exists(self, file_path):
        dir_name = os.path.dirname(file_path)
        if not os.path.exists(dir_name) and dir_name: # Check dir_name is not empty
            os.makedirs(dir_name, exist_ok=True)
            print(f"Created directory: {dir_name}")

    def _load_index_and_map(self):
        loaded_map = False
        if os.path.exists(self.id_map_file_path):
            try:
                with open(self.id_map_file_path, 'r', encoding='utf-8') as f:
                    self.faiss_ids_map = [line.strip() for line in f.readlines() if line.strip()]
                print(f"FAISS ID map loaded from {self.id_map_file_path} with {len(self.faiss_ids_map)} entries.")
                loaded_map = True
            except Exception as e:
                print(f"Error loading FAISS ID map from {self.id_map_file_path}: {e}. ID map will be empty/rebuilt.")
                self.faiss_ids_map = []
        else:
            print(f"FAISS ID map file not found at {self.id_map_file_path}.")
            self.faiss_ids_map = []

        if os.path.exists(self.index_file_path):
            print(f"Loading FAISS index from {self.index_file_path}...")
            try:
                self.index = faiss.read_index(str(self.index_file_path))
                print(f"FAISS index loaded. Contains {self.index.ntotal} vectors.")
                if self.index.ntotal != len(self.faiss_ids_map) and loaded_map: # Only warn if map was successfully loaded
                    print(f"Warning: FAISS index ({self.index.ntotal}) and loaded ID map ({len(self.faiss_ids_map)}) size mismatch.")
                elif not loaded_map and self.index.ntotal > 0:
                     print(f"Warning: Index loaded with {self.index.ntotal} vectors, but no ID map was found/loaded. IDs will be inconsistent until rebuild/load.")
                     self.faiss_ids_map = [f"temp_id_{i}" for i in range(self.index.ntotal)] # Placeholder IDs
            except Exception as e:
                print(f"Error loading FAISS index from {self.index_file_path}: {e}. Index will be None/rebuilt.")
                self.index = None
                self.faiss_ids_map = [] # Reset map if index load fails
        else:
            print(f"FAISS index file not found at {self.index_file_path}. A new index will be created upon build() or add().")
            self.index = None
            self.faiss_ids_map = []

    def _save_index_and_map(self):
        if self.index is not None:
            self._ensure_dir_exists(self.index_file_path)
            print(f"Saving FAISS index to {self.index_file_path} with {self.index.ntotal} vectors...")
            try:
                faiss.write_index(self.index, str(self.index_file_path))
                print("FAISS index saved successfully.")
                
                # Save the ID map
                with open(self.id_map_file_path, 'w', encoding='utf-8') as f:
                    for item_id in self.faiss_ids_map:
                        f.write(f"{item_id}\n")
                print(f"FAISS ID map saved to {self.id_map_file_path} with {len(self.faiss_ids_map)} entries.")
            except Exception as e:
                print(f"Error saving FAISS index or ID map: {e}")
        else:
            print("No FAISS index to save (index is None).")

    def build(self, ids: List[str], vecs: np.ndarray):
        if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
            raise ValueError(f"Input vectors must be 2D with dimension {self.dimension}, got {vecs.shape}")
        if len(ids) != vecs.shape[0]:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({vecs.shape[0]})")

        print(f"Building new FAISS index with {vecs.shape[0]} vectors.")
        # Using IndexFlatIP (Inner Product) as it's common for cosine similarity with normalized embeddings
        self.index = faiss.IndexFlatIP(self.dimension) 
        self.faiss_ids_map = [] # Reset map for a fresh build
        self.add(ids, vecs) # add will handle saving

    def add(self, ids: List[str], vecs: np.ndarray):
        if self.index is None:
            print("FAISS index not initialized. Creating a default IndexFlatIP for adding data.")
            self.index = faiss.IndexFlatIP(self.dimension)
            self.faiss_ids_map = [] # Ensure map is also new

        if vecs.ndim != 2 or vecs.shape[1] != self.dimension:
            raise ValueError(f"Input vectors must be 2D with dimension {self.dimension}, got {vecs.shape}")
        if len(ids) != vecs.shape[0]:
            raise ValueError(f"Number of IDs ({len(ids)}) must match number of vectors ({vecs.shape[0]})")

        print(f"Adding {vecs.shape[0]} vectors to FAISS index...")
        self.index.add(vecs.astype(np.float32))
        self.faiss_ids_map.extend(ids)
        self._save_index_and_map()
        print(f"FAISS index now contains {self.index.ntotal} vectors. ID map size: {len(self.faiss_ids_map)}.")

    def search(self, vec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if self.index is None or self.index.ntotal == 0:
            print("FAISS index is not initialized or is empty. Cannot search.")
            return [], []
        
        if not self.faiss_ids_map:
            print("Warning: FAISS ID map is empty. Search results will lack original IDs.")
            # Fallback or error based on desired behavior
        elif self.index.ntotal != len(self.faiss_ids_map):
            print(f"Warning: Mismatch between index size ({self.index.ntotal}) and ID map size ({len(self.faiss_ids_map)}). Search results might be incorrect or incomplete.")

        if vec.ndim == 1:
            query_vecs = np.array([vec], dtype=np.float32)
        elif vec.ndim == 2 and vec.shape[0] == 1:
            query_vecs = vec.astype(np.float32)
        else:
            raise ValueError(f"Query vector must be 1D or a single 2D vector, got shape {vec.shape}")

        effective_k = min(k, self.index.ntotal)
        if effective_k == 0:
            return [], []

        # print(f"Searching in FAISS for {effective_k} nearest neighbors...")
        distances, faiss_indices = self.index.search(query_vecs, effective_k)
        
        result_ids = []
        result_distances = []
        
        if not self.faiss_ids_map: # If map is empty, can't return string IDs
            print("Cannot map FAISS indices to string IDs because ID map is empty.")
            return [str(fi) for fi in faiss_indices[0] if fi >=0], distances[0].tolist()


        for i, faiss_idx in enumerate(faiss_indices[0]):
            if 0 <= faiss_idx < len(self.faiss_ids_map):
                result_ids.append(self.faiss_ids_map[faiss_idx])
                result_distances.append(float(distances[0][i]))
            else:
                # This case should ideally not happen if k <= ntotal and map is correct
                print(f"Warning: Invalid FAISS index {faiss_idx} encountered during search result mapping (ID map size: {len(self.faiss_ids_map)}).")
        
        return result_ids, result_distances

    def count(self) -> int:
        if self.index:
            return self.index.ntotal
        return 0

    def dump(self, dump_base_path_str: str):
        """Dumps the current index and its ID map to the specified base path."""
        if self.index is None or not self.faiss_ids_map:
            print("Nothing to dump: FAISS index is None or ID map is empty.")
            return

        dump_index_file = Path(dump_base_path_str)
        dump_map_file = dump_index_file.with_suffix(dump_index_file.suffix + ".map")
        
        self._ensure_dir_exists(dump_index_file)
        print(f"Dumping FAISS index to {dump_index_file} ({self.index.ntotal} vectors) and map to {dump_map_file} ({len(self.faiss_ids_map)} IDs)...")
        try:
            faiss.write_index(self.index, str(dump_index_file))
            with open(dump_map_file, 'w', encoding='utf-8') as f:
                for item_id in self.faiss_ids_map:
                    f.write(f"{item_id}\n")
            print("Dump successful.")
        except Exception as e:
            print(f"Error during FAISS dump: {e}")

    def load(self, load_base_path_str: str) -> bool:
        """Loads an index and its ID map from the specified base path, replacing the current one."""
        load_index_file = Path(load_base_path_str)
        load_map_file = load_index_file.with_suffix(load_index_file.suffix + ".map")

        print(f"Attempting to load FAISS index from {load_index_file} and map from {load_map_file}...")
        
        temp_ids_map = []
        if os.path.exists(load_map_file):
            try:
                with open(load_map_file, 'r', encoding='utf-8') as f:
                    temp_ids_map = [line.strip() for line in f.readlines() if line.strip()]
                print(f"Successfully read ID map file with {len(temp_ids_map)} entries.")
            except Exception as e:
                print(f"Error reading ID map file {load_map_file}: {e}. Cannot load.")
                return False # Indicate failure
        else:
            print(f"ID map file {load_map_file} not found. Cannot load.")
            return False

        if os.path.exists(load_index_file):
            try:
                temp_index = faiss.read_index(str(load_index_file))
                print(f"Successfully read FAISS index file with {temp_index.ntotal} vectors.")
                
                if temp_index.ntotal != len(temp_ids_map):
                    print(f"Warning: Loaded index vector count ({temp_index.ntotal}) "
                          f"does not match ID map entry count ({len(temp_ids_map)}). "
                          "Proceeding with load, but there might be inconsistencies.")

                self.index = temp_index
                self.faiss_ids_map = temp_ids_map
                # Update internal paths to reflect that we've loaded from this new source
                # Or decide if load() should also update self.index_file_path etc.
                # For now, it just loads into memory. The main configured paths remain.
                print("FAISS index and ID map loaded successfully into memory.")
                return True # Indicate success
            except Exception as e:
                print(f"Error reading FAISS index file {load_index_file}: {e}. Load failed.")
                return False
        else:
            print(f"FAISS index file {load_index_file} not found. Cannot load.")
            return False 