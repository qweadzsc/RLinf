#!/bin/bash

set -ex

# Build step 1: set your wiki_dataset corpus path
WIKI2018_DIR="${WIKI2018_DIR:-/path/to/wiki_dataset}"
corpus_file=$WIKI2018_DIR/wiki_corpus.jsonl

# Build step 2: set your retriever model path
retriever_name=e5 # this is for indexing naming
retriever_path="${RETRIEVER_PATH:-/path/to/retriever/model}"

# Qdrant server configuration
qdrant_url="${QDRANT_URL:-http://localhost:6333}"
qdrant_collection_name="${QDRANT_COLLECTION_NAME:-wiki_collection}"
hnsw_config='{"m":32,"ef_construct":512}'
retrieval_batch_size="${RETRIEVAL_BATCH_SIZE:-128}"
resume_args=""
if [[ "${RESUME:-0}" == "1" ]]; then
    resume_args="--resume"
fi
if [[ -n "${RESUME_OVERLAP:-}" ]]; then
    resume_args="${resume_args} --resume_overlap ${RESUME_OVERLAP}"
fi

CONFIG_PATH="$( realpath "$( dirname "${BASH_SOURCE[0]}" )"  )"
ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}" \
python3 "${CONFIG_PATH}/build_index_ascend.py" \
    --corpus_path "${corpus_file}" \
    --retriever_name "${retriever_name}" \
    --retriever_model "${retriever_path}" \
    --qdrant_collection_name "${qdrant_collection_name}" \
    --qdrant_url "${qdrant_url}" \
    --hnsw_config "${hnsw_config}" \
    --build_parallel "${BUILD_PARALLEL:-8}" \
    --retrieval_batch_size "${retrieval_batch_size}" \
    ${resume_args}
