import time

from transformers import pipeline

from PIL import Image

image = Image.open('TelegramMessages.png').convert("RGB")

start_time = time.perf_counter()

image_text_to_text_converter = pipeline("image-text-to-text", "01-ai/Yi-VL-34B", device_map=2)
generated_text = image_text_to_text_converter(image)

print(generated_text)
end_time = time.perf_counter()
print("Run Time: " + (end_time - start_time).__str__() + "s")
