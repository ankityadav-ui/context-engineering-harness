from pathlib import Path

from ollama import chat


class LLMError(Exception):
    """Raised when the LLM cannot generate a response."""


def generate_response(message: str) -> str:
    """
    Generate a normal text-only response using llama3.2.
    """
    try:
        response = chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
        )

        return response.message.content

    except Exception as error:
        print("LLM ERROR:", repr(error))

        raise LLMError(
            f"The language model could not generate a response: {error}"
        ) from error


def generate_multimodal_response(
    message: str,
    image_paths: list[str],
) -> str:
    """
    Generate a response using text plus one or more images.
    The Qwen multimodal model receives the actual image files.
    """
    try:
        valid_images = []

        for image_path in image_paths:
            path = Path(image_path)

            print("IMAGE PATH:", path)
            print("IMAGE EXISTS:", path.exists())

            if path.exists():
                print("IMAGE SIZE:", path.stat().st_size)
                valid_images.append(str(path))

        if not valid_images:
            raise ValueError(
                "No valid image files were found."
            )

        response = chat(
            model="qwen3.5:4b",
            messages=[
                {
                    "role": "user",
                    "content": message,
                    "images": valid_images,
                }
            ],
        )

        return response.message.content

    except Exception as error:
        print("MULTIMODAL ERROR:", repr(error))

        raise LLMError(
            f"Multimodal model error: {error}"
        ) from error