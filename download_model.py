from huggingface_hub import hf_hub_download


# Hugging Face repository containing the trained model
HF_REPO_ID = "urjit006/GPiSeT"

MODEL_FILENAME = "GPiSeT-CNN.keras" #change to GPiSeT-H for model trained with heuristic guidance


def download_model():
    """
    Download the pretrained model from Hugging Face.

    Returns:
        str: Local path to the downloaded model.
    """

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=MODEL_FILENAME
    )

    print(f"\nModel downloaded successfully:")
    print(model_path)

    return model_path


if __name__ == "__main__":
    download_model()
