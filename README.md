## Installation

You can clone the repository and install it with `pip install .``

### Using Geneformer

If you plan to use geneformer, it might not work with all python versions. For me this works:

First make sure the external repo is initialised:
To use the Geneformer submodule, you need to have git lfs installed. Run 'git lfs install' if you have it installed.
Then, run 'git submodule update --init --recursive' to update the geneformer external repository
You can try: 'pip install external/Geneformer'. But I've had issues with python versions. This works for me:

install python 3.10.16 with uv:
uv python install 3.10.16
uv venv -p 3.10.16
uv pip install external/Geneformer
uv sync
