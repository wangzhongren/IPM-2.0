import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

def download_models():
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Download Phi-3-mini for memory navigation from Hugging Face Hub
    print("Downloading Phi-3-mini from Hugging Face Hub...")
    snapshot_download(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        local_dir=os.path.join(models_dir, "phi-3-mini-4k-instruct"),
        local_dir_use_symlinks=False
    )
    
    print("Models downloaded successfully!")

if __name__ == "__main__":
    download_models()