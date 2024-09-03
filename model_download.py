from huggingface_hub import snapshot_download, hf_hub_download
import os

repos = ["joaopn/glove-model-reduced-stopwords", "joaopn/unbiased-toxic-roberta-onnx-fp16"]

files = [{'repo_id':"facebook/fasttext-language-identification", 'filename': "model.bin"}]

current_dir = os.path.dirname(__file__)

for repo in repos:
    try :
        model_folder = repo.replace("/","_")
        filepath = os.path.join(current_dir, 'civirank', 'models', model_folder)
        snapshot_download(repo_id=repo,local_dir=filepath, repo_type="model")
    except Exception as e:
        raise e

for file in files:
    try :
        model_folder = file['repo_id'].replace("/","_")
        filepath = os.path.join(current_dir, 'civirank', 'models', model_folder)
        hf_hub_download(repo_id=file['repo_id'], filename=file['filename'], local_dir=filepath)
    except Exception as e:
        raise e

print("Downloaded models successfully")