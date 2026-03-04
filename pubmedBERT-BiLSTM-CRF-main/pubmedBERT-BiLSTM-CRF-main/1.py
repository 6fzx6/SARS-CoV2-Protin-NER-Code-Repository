
from transformers import AutoTokenizer, AutoModel, AutoConfig

model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
cache_dir = "./pretrained_models"

# 下载并缓存模型文件
config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

print("Model downloaded successfully!")
