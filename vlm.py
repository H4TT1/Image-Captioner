import torch
import csv
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoImageProcessor, VisionEncoderDecoderModel
from tqdm import tqdm
import time


MODEL_PATH = "cnmoro/tiny-image-captioning"
TEST_IMG_DIR = Path("./data/processed/test/")
TEST_CAPTIONS_CSV = Path("./data/processed/test_captions.csv")
OUTPUT_CSV = Path("./vlm_predictions.csv")


print("Loading VLM model...")
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
image_processor = AutoImageProcessor.from_pretrained(MODEL_PATH)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


print("Loading test captions...")
test_images = set()
with open(TEST_CAPTIONS_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='|')
    for row in reader:
        test_images.add(row['image'])

test_images = sorted(list(test_images))
print(f"Found {len(test_images)} unique test images")


# If .pt files are normalized tensors, we need to denormalize them
# Standard ImageNet normalization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_tensor(tensor):
    """Denormalize tensor back to [0, 1] range"""
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # tensor: [3, H, W]
    tensor = denormalize_tensor(tensor)
    # Convert to [H, W, 3] and scale to [0, 255]
    img_array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
    return Image.fromarray(img_array)


print("\nGenerating captions...")
results = []
total_time = 0
failed_images = []

with torch.no_grad():
    for img_name in tqdm(test_images, desc="Processing"):
        try:
            # Load .pt file
            pt_path = TEST_IMG_DIR / img_name.replace('.jpg', '.pt')
            
            if not pt_path.exists():
                print(f"\nWarning: {pt_path} not found, skipping...")
                failed_images.append(img_name)
                continue
            
            # Load tensor and convert to PIL
            img_tensor = torch.load(pt_path)
            pil_image = tensor_to_pil(img_tensor)
            
            # Preprocess with VLM's image processor
            pixel_values = image_processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            # Generate caption
            start = time.time()
            generated_ids = model.generate(
                pixel_values,
                temperature=0.7,
                top_p=0.8,
                top_k=50,
                num_beams=3,
                max_length=50
            )
            end = time.time()
            
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            inference_time = end - start
            total_time += inference_time
            
            results.append({
                'image': img_name,
                'predicted_caption': generated_text,
                'inference_time': f"{inference_time:.3f}"
            })
            
        except Exception as e:
            print(f"\nError processing {img_name}: {e}")
            failed_images.append(img_name)
            continue


print(f"\nSaving predictions to {OUTPUT_CSV}...")
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    fieldnames = ['image', 'predicted_caption', 'inference_time']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print("\n" + "="*50)
print("INFERENCE COMPLETE")
print("="*50)
print(f"Total images processed: {len(results)}")
print(f"Failed images: {len(failed_images)}")
print(f"Total time: {total_time:.2f} seconds")
print(f"Average time per image: {total_time/len(results):.3f} seconds")
print(f"Output saved to: {OUTPUT_CSV}")

if failed_images:
    print(f"\nFailed images ({len(failed_images)}):")
    for img in failed_images[:10]:  
        print(f"  - {img}")
    if len(failed_images) > 10:
        print(f"  ... and {len(failed_images) - 10} more")


print("\n" + "="*50)
print("SAMPLE PREDICTIONS")
print("="*50)
for i, result in enumerate(results[:5]):
    print(f"\n{i+1}. Image: {result['image']}")
    print(f"   Caption: {result['predicted_caption']}")
    print(f"   Time: {result['inference_time']}s")