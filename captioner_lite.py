from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

class CaptionerLite:
    def __init__(self):
        self.model_name = "nlpconnect/vit-gpt2-image-captioning"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_caption(self, image: Image.Image) -> str:
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(pixel_values, max_length=16, num_beams=2)
                caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return caption
        except Exception as e:
            print(f"Captioning failed: {e}")
            return ""
