from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="psuplj/Meta-Llama-3-8B-Q4_K_M-GGUF",
    local_dir="models/tokenizer",
    allow_patterns=["*.json", "*.model", "*.py", "*.txt"]
)