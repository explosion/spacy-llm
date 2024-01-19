from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

while True:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, resume_download=True)
        break
    except Exception:
        pass
