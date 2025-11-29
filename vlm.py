from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
from PIL import Image
import time

model_path = "cnmoro/tiny-image-captioning"

# load the model, tokenizer and processor
model = VisionEncoderDecoderModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
image_processor = AutoImageProcessor.from_pretrained(model_path)

# ---- LOAD IMAGE FROM LOCAL DISK ----
image_path = "/home/ihatti/MLnAI/enseirb/computer-vision/computer-vision/image-capt-project/data/images/3737539561_d1dc161040.jpg"   # <-- put your local image path here
image = Image.open(image_path).convert("RGB")

# preprocess
pixel_values = image_processor(image, return_tensors="pt").pixel_values

start = time.time()

# generate caption
generated_ids = model.generate(
    pixel_values,
    temperature=0.7,
    top_p=0.8,
    top_k=50,
    num_beams=3
)

generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

end = time.time()

print(generated_text)
print(f"Time taken: {end - start:.3f} seconds")
