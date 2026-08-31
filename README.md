# FocusLens

An interactive ImageNet image-classification workbench built with FastAPI. It uses **only ResNet18 pretrained on ImageNet-1K** for classification and a self-contained PyTorch autograd implementation of **Integrated Gradients** for explanations.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn server:app --reload
```

Open `http://127.0.0.1:8000`. The first request downloads the official ResNet18 ImageNet-1K weights if they are not already cached.

## Using the explanation

Upload any compatible image, inspect the top five ImageNet predictions, and select any shown class to explain it. The app calculates all four references together: neutral, black, white, and blurred. Orange regions support the selected class, while blue regions oppose it. Compare the maps: consistent focus is a stronger sign that the explanation is robust; major changes expose baseline sensitivity.
