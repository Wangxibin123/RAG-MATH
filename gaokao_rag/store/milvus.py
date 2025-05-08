from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from .base import BaseStore
from ..cfg import CFG
import numpy as np
from typing import List, Tuple

class MilvusStore(BaseStore):
    def __init__(self):
        self.param = CFG.store
        self.collection_name = self.param['collection']
        self.alias = self.param.get('alias', 'default') # Use alias from config or default

        # Connect to Milvus
        try:
            print(f"Connecting to Milvus: host={self.param['host']}, port={self.param['port']}, alias={self.alias}")
            connections.connect(
                alias=self.alias,
                host=self.param['host'],
                port=str(self.param['port'])
            )
            print(f"Successfully connected to Milvus with alias '{self.alias}'.")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            raise

        # Get or create collection
        if utility.has_collection(self.collection_name, using=self.alias):
            print(f"Collection '{self.collection_name}' exists. Loading...")
            self.col = Collection(self.collection_name, using=self.alias)
        else:
            print(f"Collection '{self.collection_name}' does not exist. Creating...")
            self.col = self._create_collection(self.param)
        
        # Load collection before search, can be done once at init if data doesn't change often
        # Or, ensure it's loaded before each search if that's more appropriate.
        print(f"Loading collection '{self.collection_name}' into memory...")
        self.col.load()
        print(f"Collection '{self.collection_name}' loaded.")

    def _create_collection(self, p):
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64, description="Primary key ID"),
            FieldSchema(name="vec", dtype=DataType.FLOAT_VECTOR, dim=CFG.embed_dim, description="Float vector embedding")
        ]
        schema = CollectionSchema(fields, description=f"Collection for {self.collection_name}")
        print(f"Creating collection '{self.collection_name}' with schema: {fields}")
        col = Collection(self.collection_name, schema=schema, using=self.alias)
        
        index_params = {
            "metric_type": "IP",  # Inner Product for similarity
            "index_type": "HNSW",
            "params": {"M": p['params']['M'], "efConstruction": p['params']['efConstruction']}
        }
        print(f"Creating index for 'vec' field with params: {index_params}")
        col.create_index(field_name="vec", index_params=index_params)
        print("Index created.")
        return col

    def build(self, ids: List[str], vecs: np.ndarray):
        # For Milvus, build is often part of initial creation or a large batch insert.
        # If collection exists and has data, we might want to clear it first for a true 'build'.
        if utility.has_collection(self.collection_name, using=self.alias):
            # Check if there's data. If so, decide on dropping or simply adding.
            # For a true `build` from scratch, one might drop and recreate.
            # self.col.drop_index() # if needed
            # self.col.release() # if needed
            # utility.drop_collection(self.collection_name, using=self.alias) # if needed
            # self.col = self._create_collection(self.param)
            print(f"Collection '{self.collection_name}' exists. Adding data to it.")
        
        print(f"Building index by adding {len(ids)} vectors.")
        self.add(ids, vecs) # Milvus HNSW index is built incrementally

    def add(self, ids: List[str], vecs: np.ndarray):
        if not ids or vecs.size == 0:
            print("No data provided to add.")
            return
        
        data = [ids, vecs.tolist()] # Milvus expects list of lists for insert
        print(f"Inserting {len(ids)} entities into '{self.collection_name}'...")
        try:
            mr = self.col.insert(data)
            print(f"Insertion result: {mr}")
            print("Flushing collection to ensure data persistence...")
            self.col.flush()
            print("Flush complete.")
        except Exception as e:
            print(f"Error during Milvus insert/flush: {e}")
            raise

    def search(self, vec: np.ndarray, k: int) -> Tuple[List[str], List[float]]:
        if vec.ndim == 1:
            search_vecs = [vec.tolist()] # Search expects a list of vectors
        else:
            search_vecs = vec.tolist()

        search_params = {
            "metric_type": "IP",
            "params": {"ef": CFG.store['params']['efSearch']}
        }
        # print(f"Searching with vector: {search_vecs[0][:5]}... (first 5 dims), k={k}, params={search_params}")
        
        # Ensure collection is loaded (might be redundant if loaded at init and stays loaded)
        # self.col.load() 
        
        results = self.col.search(
            data=search_vecs,
            anns_field="vec",
            param=search_params,
            limit=k,
            expr=None, # No filtering expression for now
            output_fields=['id'], # Request 'id' to be returned
            consistency_level="Strong" # Or other levels like "Bounded", "Eventually"
        )
        
        # print(f"Search raw results: {results}")
        
        # Process results for a single query vector input
        # Milvus search returns a list of hit lists, one for each query vector.
        if not results or not results[0]:
            return [], []

        hit_list = results[0]
        result_ids = [hit.id for hit in hit_list]
        result_distances = [hit.distance for hit in hit_list]
        
        # print(f"Search found IDs: {result_ids}, Distances: {result_distances}")
        return result_ids, result_distances

    def count(self):
        """Returns the number of entities in the collection."""
        return self.col.num_entities

    # Consider adding a disconnect method if needed for graceful shutdown
    # def disconnect(self):
    #     connections.disconnect(self.alias)
    #     print(f"Disconnected from Milvus alias '{self.alias}'.") 