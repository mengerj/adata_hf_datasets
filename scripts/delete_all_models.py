from huggingface_hub import HfApi

# Initialize API
api = HfApi()

# Get the list of models for the authenticated user
user = api.whoami()
username = user["name"]

models = api.list_models(author=username)
models_list = list(models)

if not models_list:
    print("No models found.")
else:
    print(f"Found {len(models_list)} models. Deleting...")

    for model in models_list:
        model_id = model.modelId
        print(f"Deleting model: {model_id}")
        api.delete_repo(repo_id=model_id, repo_type="model")

    print("All models deleted successfully!")
