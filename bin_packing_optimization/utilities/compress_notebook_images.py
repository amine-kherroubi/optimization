"""
compress_notebook.py

Usage: python compress_notebook.py input.ipynb [output.ipynb]

Reduces the width of embedded images in a Jupyter notebook while keeping PNG format.

Requires: pip install pillow
"""

import sys
import json
import base64
import io
from PIL import Image

MAX_WIDTH = 900


def resize_image(b64_data: str) -> str:
    img = Image.open(io.BytesIO(base64.b64decode(b64_data)))

    if img.width <= MAX_WIDTH:
        return b64_data

    new_height = int(img.height * MAX_WIDTH / img.width)
    resample = getattr(Image, "Resampling", None)
    resample = resample.LANCZOS if resample else Image.BICUBIC  # type: ignore
    img = img.resize((MAX_WIDTH, new_height), resample)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def compress_notebook(input_path: str, output_path: str) -> None:
    with open(input_path) as f:
        nb = json.load(f)

    original_size = 0
    new_size = 0
    image_count = 0

    for cell in nb["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime in list(data.keys()):
                if "image/png" not in mime and "image/jpeg" not in mime:
                    continue

                raw = data[mime] if isinstance(data[mime], str) else "".join(data[mime])
                original_size += len(raw) * 3 // 4

                resized = resize_image(raw)
                new_size += len(resized) * 3 // 4

                data[mime] = resized
                image_count += 1

    with open(output_path, "w") as f:
        json.dump(nb, f, indent=1)

    savings = (1 - new_size / original_size) * 100 if original_size else 0
    print(f"Images processed : {image_count}")
    print(
        f"Image data       : {original_size/1024:.1f} KB → {new_size/1024:.1f} KB ({savings:.1f}% savings)"
    )
    print(f"Saved to         : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_notebook.py input.ipynb [output.ipynb]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path

    compress_notebook(input_path, output_path)
