import time
import numpy as np

from transformers import pipeline
from PIL import Image

vqa = pipeline(model="impira/layoutlm-document-qa")

output = vqa(image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png", question="What is the invoice number?")

# image = Image.open('TelegramMessages.png')
#
start_time = time.perf_counter()
# image_to_text_converter = pipeline("image-to-text", "Salesforce/blip-image-captioning-large", device=2)
# output = image_to_text_converter(image, )
# task="Please output the text that is shown in this image."
print(output)

end_time = time.perf_counter()

print("Run Time: " + (end_time - start_time).__str__() + "s")
