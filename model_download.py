from huggingface_hub import snapshot_download

repos = ["facebook/fasttext-language-identification", "joaopn/glove-model-reduced-stopwords", "joaopn/unbiased-toxic-roberta-onnx-fp16"]

for repo in repos:
    try :
        local_dir = repo.replace("/","_")
        snapshot_download(repo_id=repo,local_dir="models/"+local_dir, repo_type="model")
    except Exception as e:
        raise e

print("Downloaded models successfully")