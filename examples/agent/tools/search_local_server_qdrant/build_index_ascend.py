# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import queue
import time
import warnings
from typing import Any, Optional

import datasets
import torch
import torch_npu  # noqa: F401
from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionStatus,
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)
from qdrant_encoder import Encoder
from tqdm import tqdm

global_encoder = None
global_client = None


def get_npu_device(index: int = 0) -> torch.device:
    """Return an available NPU device for an encoder worker."""
    if not torch.npu.is_available():
        raise RuntimeError("No NPU device is available for Ascend index building")
    device_count = torch.npu.device_count()
    if device_count < 1:
        raise RuntimeError("No NPU device is available for Ascend index building")
    return torch.device(f"npu:{index % device_count}")


def set_global(retrieval_method, config):
    from multiprocessing import current_process

    process_idx = current_process()._identity[0]

    global global_encoder
    global_encoder = Encoder(
        model_name=retrieval_method,
        model_path=config.retrieval_model_path,
        pooling_method=config.retrieval_pooling_method,
        max_length=config.retrieval_query_max_length,
        use_fp16=config.retrieval_use_fp16,
        device=get_npu_device(process_idx - 1),
        attention_implementation="eager",
    )

    global global_client
    global_client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        "json",
        data_files=corpus_path,
        split="train",
        num_proc=8,
        # cache dir can be customized into path/to/your/cache/dir
        cache_dir="~/.cache",
    )
    return corpus


def read_jsonl(file_path: str) -> list[Any]:
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


class QdrantIndexBuilder:
    def __init__(self, config):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)
        self.collection_name = config.qdrant_collection_name

    def build(self):
        # Initialize encoder first (needed for building collection)
        self.topk = self.config.retrieval_topk
        self.batch_size = self.config.retrieval_batch_size

        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        existing_document_count = 0
        if self.collection_name in collection_names and self.config.resume:
            existing_document_count = self.client.count(
                collection_name=self.collection_name, exact=True
            ).count
            logging.info(
                "Resuming collection '%s' from %d existing points.",
                self.collection_name,
                existing_document_count,
            )
        elif self.collection_name in collection_names:
            logging.info(
                f"Collection '{self.collection_name}' already exists. deleting..."
            )
            self.client.delete_collection(collection_name=self.collection_name)
        else:
            logging.info(f"Collection '{self.collection_name}' not found.")
        logging.info("Building collection from corpus...")
        self._build_collection_from_corpus(
            self.config.corpus_text_field, existing_document_count
        )
        logging.info(f"Collection '{self.collection_name}' built successfully!")

    @staticmethod
    def encode_and_upsert(config, batch_texts, batch_indices, batch_payload):
        collection_name = config.qdrant_collection_name
        global global_encoder
        global global_client

        points = []
        batch_emb = global_encoder.encode(batch_texts, is_query=False)
        # Create points
        for emb, doc_idx, payload in zip(batch_emb, batch_indices, batch_payload):
            points.append(
                PointStruct(
                    id=doc_idx,
                    vector=emb.tolist(),
                    payload=payload,
                )
            )
        global_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def _build_collection_from_corpus(self, text_field=None, existing_document_count=0):
        """Build Qdrant collection from corpus."""
        corpus_data = load_corpus(self.config.corpus_path)
        corpus_size = len(corpus_data)
        logging.info(f"Corpus size: {corpus_size} documents")

        # Get vector dimension by encoding a sample document
        sample_text = "hello, world!"
        encoder = Encoder(
            model_name=self.config.retrieval_method,
            model_path=self.config.retrieval_model_path,
            pooling_method=self.config.retrieval_pooling_method,
            max_length=self.config.retrieval_query_max_length,
            use_fp16=self.config.retrieval_use_fp16,
            device=get_npu_device(),
            attention_implementation="eager",
        )
        sample_emb = encoder.encode(sample_text, is_query=False)
        vector_size = sample_emb.shape[1]
        encoder = None
        logging.info(f"Vector dimension: {vector_size}")

        # Create collection
        try:
            hnsw_config = json.loads(self.config.hnsw_config)
            logging.info(f"hnsw_config uses: {hnsw_config}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(**hnsw_config),
            )
        except Exception as e:
            # Collection might already exist, check and handle
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self.collection_name in collection_names:
                logging.info(
                    f"Collection '{self.collection_name}' already exists, skipping creation."
                )
            else:
                raise e

        # Encode and insert documents in batches
        # Multiprocessing setup for parallel document processing
        from multiprocessing import Pool

        # Create a process pool with the specified number of parallel workers
        # The initializer function 'set_global' is called once per worker process
        # to initialize the global encoder and client for each worker
        pool = Pool(
            self.config.build_parallel,
            initializer=set_global,
            initargs=(self.config.retrieval_method, self.config),
        )
        handles = queue.Queue()

        batch_texts = []
        batch_indices = []
        batch_payload = []
        start_index = 0
        if self.config.resume and existing_document_count:
            default_overlap = self.config.build_parallel * 10 * self.batch_size * 2
            overlap = self.config.resume_overlap or default_overlap
            start_index = max(0, existing_document_count - overlap)
            logging.info(
                "Resuming from corpus index %d with a %d-document overlap.",
                start_index,
                existing_document_count - start_index,
            )

        expected_document_count = 0
        for idx in tqdm(
            range(start_index, corpus_size),
            desc="Building collection",
            initial=start_index,
            total=corpus_size,
        ):
            doc = corpus_data[idx]
            assert self.config.retrieval_method == "e5", (
                f"Expected retrieval_method to be 'e5', but got '{self.config.retrieval_method}'"
            )
            text = doc["contents"]

            # Skip empty texts
            if not text or len(text.strip()) == 0:
                warnings.warn(f"Document {idx} has empty text, skipping...")
                continue
            batch_texts.append(text)
            batch_indices.append(idx)
            batch_payload.append(doc)
            expected_document_count += 1

            # Process batch when it reaches batch_size
            if len(batch_texts) >= self.batch_size:
                # Submit batch for parallel processing using async execution
                # This returns a handle that can be used to track task completion
                handle = pool.apply_async(
                    QdrantIndexBuilder.encode_and_upsert,
                    (self.config, batch_texts, batch_indices, batch_payload),
                )
                handles.put(handle)

                # Control memory usage by waiting for some tasks to complete
                # if too many tasks are queued (prevents memory overflow)
                if handles.qsize() >= self.config.build_parallel * 10:
                    handles.get().get()

                # Reset batch variables for next batch
                batch_texts = []
                batch_indices = []
                batch_payload = []

        # Process remaining items
        if batch_texts:
            handle = pool.apply_async(
                QdrantIndexBuilder.encode_and_upsert,
                (self.config, batch_texts, batch_indices, batch_payload),
            )
            handles.put(handle)

        # Wait for all remaining tasks to complete
        # This ensures all documents are processed before closing the pool
        while not handles.empty():
            handles.get().get()

        pool.close()
        pool.join()

        # Wait for the collection to be ready (GREEN status)
        # This ensures all data has been properly indexed and is available for querying
        logging.info("wait collection status to be green")
        while (
            self.client.get_collection(self.collection_name).status
            != CollectionStatus.GREEN
        ):
            time.sleep(1)
        collection_info = self.client.get_collection(self.collection_name)
        actual_document_count = self.client.count(
            collection_name=self.collection_name, exact=True
        ).count
        expected_final_count = (
            corpus_size if self.config.resume else expected_document_count
        )
        logging.info(
            "collection status of '%s' is %s; expected=%d, actual=%d",
            self.collection_name,
            collection_info.status,
            expected_final_count,
            actual_document_count,
        )
        if actual_document_count != expected_final_count:
            raise RuntimeError(
                "Qdrant point-count mismatch after build: "
                f"expected {expected_final_count}, got {actual_document_count}"
            )
        logging.info(
            "Successfully inserted %d non-empty documents into collection '%s'",
            expected_final_count,
            self.collection_name,
        )


class Config:
    """
    Minimal config class for Qdrant index builder (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        qdrant_url: Optional[str] = None,
        qdrant_collection_name: str = "default_collection",
        corpus_text_field: Optional[str] = None,
        hnsw_config: str | None = None,
        build_parallel: int | None = None,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        resume: bool = False,
        resume_overlap: int | None = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.corpus_text_field = corpus_text_field
        self.hnsw_config = hnsw_config
        self.build_parallel = build_parallel
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.resume = resume
        self.resume_overlap = resume_overlap


if __name__ == "__main__":
    from multiprocessing import set_start_method

    set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Build a local Qdrant index on Ascend NPUs."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Local corpus file.",
    )
    parser.add_argument(
        "--retriever_name", type=str, default="e5", help="Name of the retriever model."
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Path of the retriever model.",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=None,
        help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.",
    )
    parser.add_argument(
        "--qdrant_collection_name",
        type=str,
        default="default_collection",
        help="Name of the Qdrant collection.",
    )
    parser.add_argument(
        "--corpus_text_field",
        type=str,
        default=None,
        help="Field name in corpus documents containing text to encode. If not specified, will try common field names (text, contents, passage, etc.).",
    )
    parser.add_argument(
        "--hnsw_config", type=str, default="", help="Qdrant hnsw config"
    )
    parser.add_argument(
        "--build_parallel", type=int, default=8, help="Qdrant build thread"
    )
    parser.add_argument(
        "--retrieval_batch_size",
        type=int,
        default=128,
        help="Documents encoded per NPU forward pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve an existing collection and resume with an overlapping range.",
    )
    parser.add_argument(
        "--resume_overlap",
        type=int,
        default=None,
        help="Documents to re-upsert before the estimated resume point.",
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        corpus_path=args.corpus_path,
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.qdrant_collection_name,
        corpus_text_field=args.corpus_text_field,
        hnsw_config=args.hnsw_config,
        build_parallel=args.build_parallel,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=args.retrieval_batch_size,
        resume=args.resume,
        resume_overlap=args.resume_overlap,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    index_builder = QdrantIndexBuilder(config)
    index_builder.build()
