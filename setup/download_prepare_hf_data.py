import argparse
import os
import time
import subprocess
import requests
from huggingface_hub import snapshot_download

def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

def download_dataset(repo_id, local_dir, allow_patterns):
    print(f"Downloading dataset from {repo_id}...")
    max_retries = 5
    retry_delay = 10  # seconds
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                resume_download=True,
                max_workers=16,  # Increase to lower download time if needed
            )
            break
        except requests.exceptions.ReadTimeout:
            if attempt < max_retries - 1:
                print(f"Timeout occurred. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise
    print(f"Dataset downloaded to {local_dir}")

def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                src_dir,
                file_progress=True,
                doc_progress=True,
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                tgt_dir,
                output_filename=dataset + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()

def setup_terashuf(work_dir):
    terashuf_dir = os.path.join(work_dir, "terashuf")
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    if os.path.exists(terashuf_executable):
        print("terashuf executable already exists. Skipping setup.")
        return terashuf_dir

    print("Setting up terashuf...")
    run_command(f"git clone https://github.com/alexandres/terashuf {terashuf_dir}")
    run_command(f"make -C {terashuf_dir}")
    return terashuf_dir

def main(dataset, memory, data_dir, seed=42, nchunks=32):
    # Mapping from dataset names to repository IDs.
    repo_id = {
        "fineweb_edu": "HuggingFaceFW/fineweb-edu",
        "fineweb_edu_10bt": "HuggingFaceFW/fineweb-edu",
        "dclm_baseline_1.0": "mlfoundations/dclm-baseline-1.0",
        "dclm_baseline_1.0_10prct": "mlfoundations/dclm-baseline-1.0",
        "opc_fineweb_code_corpus": "OpenCoder-LLM/opc-fineweb-code-corpus",
        "opc_sft_stage2": "OpenCoder-LLM/opc-sft-stage2",
        "opc_sft_stage1": "OpenCoder-LLM/opc-sft-stage1",
        "opc_fineweb_math_corpus": "OpenCoder-LLM/opc-fineweb-math-corpus",
        "opc_annealing_corpus": "OpenCoder-LLM/opc-annealing-corpus",
    }[dataset]

    src_dir = f"{data_dir}/{dataset}"
    out_dir = f"{src_dir}_shuffled"
    os.makedirs(out_dir, exist_ok=True)
    work_dir = src_dir  # Using the dataset directory as the working directory

    prefix = f"{dataset}.chunk."
    # File extensions for each dataset.
    orig_extension = {
        "fineweb_edu": ".jsonl",
        "fineweb_edu_10bt": ".jsonl",
        "dclm_baseline_1.0": ".jsonl.zst",
        "dclm_baseline_1.0_10prct": ".jsonl.zst",
        "opc_fineweb_code_corpus": ".parquet",
        "opc_sft_stage2": ".parquet",
        "opc_sft_stage1": ".parquet",
        "opc_fineweb_math_corpus": ".parquet",
        "opc_annealing_corpus": ".parquet",
    }[dataset]

    # Read command for each dataset.
    cat_command = {
        "fineweb_edu": "cat {}",
        "fineweb_edu_10bt": "cat {}",
        "dclm_baseline_1.0": "zstdcat {} && echo",
        "dclm_baseline_1.0_10prct": "zstdcat {} && echo",
        "opc_fineweb_code_corpus": "cat {}",
        "opc_sft_stage2": "cat {}",
        "opc_sft_stage1": "cat {}",
        "opc_fineweb_math_corpus": "cat {}",
        "opc_annealing_corpus": "cat {}",
    }[dataset]

    # Allowed file patterns for dataset download.
    allow_patterns = {
        "fineweb_edu": None,
        "fineweb_edu_10bt": "sample/10BT/*",
        "dclm_baseline_1.0": "*.jsonl.zst",
        "dclm_baseline_1.0_10prct": "global-shard_01_of_10/*.jsonl.zst",
        "opc_fineweb_code_corpus": "data/*.parquet",
        "opc_sft_stage2": "data/*.parquet",
        "opc_sft_stage1": "data/*.parquet",
        "opc_fineweb_math_corpus": "data/*.parquet",
        "opc_annealing_corpus": "data/*.parquet",
    }[dataset]

    k_validation = 10000  # Number of lines to extract from each chunk for validation

    # Setup terashuf for shuffling the dataset.
    terashuf_dir = setup_terashuf(work_dir)

    # Download dataset from Hugging Face Hub.
    download_dataset(repo_id, src_dir, allow_patterns)

    # List of datasets that are stored as Parquet and need conversion.
    parquet_datasets = [
        "opc_fineweb_code_corpus",
        "opc_sft_stage2",
        "opc_sft_stage1",
        "opc_fineweb_math_corpus",
        "opc_annealing_corpus",
    ]
    # Convert Parquet files to JSONL if required.
    if dataset in parquet_datasets:
        parquet_to_jsonl(dataset, work_dir, src_dir, src_dir)

    # Set environment variables for memory and seed.
    os.environ["MEMORY"] = f"{memory}"
    os.environ["SEED"] = f"{seed}"

    # Run the shuffling and splitting command using terashuf.
    terashuf_executable = os.path.join(terashuf_dir, "terashuf")
    run_command(
        f"ulimit -n 100000 && "
        f"find {src_dir} -type f -name '*{orig_extension}' -print0 | xargs -0 -I {{}} sh -c '{cat_command}' | {terashuf_executable} | "
        f"split -n r/{nchunks} -d --suffix-length 2 --additional-suffix .jsonl - {out_dir}/{prefix}"
        "; trap 'echo \"Caught signal 13, exiting with code 1\"; exit 1' SIGPIPE;"
    )

    # Create a validation set by extracting a fixed number of lines from each chunk.
    validation_file = f"{out_dir}/{dataset}.val.jsonl"
    for i in range(nchunks):
        chunk_file = f"{out_dir}/{prefix}{i:02d}.jsonl"
        run_command(f"head -n {k_validation} {chunk_file} >> {validation_file}")
        run_command(f"sed -i '1,{k_validation}d' {chunk_file}")

    print("All tasks completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str,
                        help="Name of the dataset. Options: fineweb_edu, fineweb_edu_10bt, dclm_baseline_1.0, dclm_baseline_1.0_10prct, opc_fineweb_code_corpus, opc_sft_stage2, opc_sft_stage1, opc_fineweb_math_corpus, opc_annealing_corpus")
    parser.add_argument("memory", type=float, help="Memory allocation (e.g., 8)")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory to store downloaded data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--nchunks", type=int, default=32, help="Number of chunks to split the dataset into")
    args = parser.parse_args()
    main(args.dataset, args.memory, args.data_dir, args.seed, args.nchunks)