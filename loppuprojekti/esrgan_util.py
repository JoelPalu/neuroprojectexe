import tensorflow_hub as hub


def get_model(model_type="esrgan"):
    if model_type == "esrgan":
        model_path = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
    else:
        raise ValueError(f"Unknown model: {model_type}")

    print(f"Loading model: {model_path}")
    return hub.load(model_path)
