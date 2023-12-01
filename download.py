from transformers import pipeline

# huggingface_hub.hf_hub_download(repo_id="HuggingFaceH4/zephyr-7b-beta")
model = pipeline(
    "text-generation", model="HuggingFaceH4/zephyr-7b-beta", resume_download=True
)
