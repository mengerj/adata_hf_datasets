# import pybiomart
from huggingface_hub import HfApi

api = HfApi()
ds_list = api.list_datasets(author="jo-mengr", limit=None)
print(len(list(ds_list)))
