from huggingface_hub import snapshot_download

repo = "joaopn/unbiased-toxic-roberta-onnx-fp16"
local_dir = repo
snapshot_download(repo_id=repo,local_dir=local_dir,repo_type="model")
