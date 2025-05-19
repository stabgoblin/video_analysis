import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import warnings
from typing import Optional, Dict, List
import yaml

class BLIP2Captioner:
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize BLIP-2 model with configuration from YAML.
        
        Args:
            config_path: Path to config.yaml
        """
        # Load configuration
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Device setup
        self.device = self._get_device()
        
        # Model initialization
        self.processor, self.model = self._load_model()
        
        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*You are using the default legacy behaviour.*")

    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config["device"] == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config["device"]

    def _load_model(self):
        model_name = "Salesforce/blip2-opt-2.7b"  
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name,torch_dtype=torch.float16,load_in_8bit=True,device_map="auto")
        return processor, model

    def generate_caption(self, image_path: str, surveillance_mode: bool = True) -> str:
        """
        Generate caption for an image with surveillance optimizations.
        
        Args:
            image_path: Path to input image
            surveillance_mode: Whether to use security-focused prompts
            
        Returns:
            Generated caption string
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Surveillance-optimized prompt
            prompt = (
                "Question: Describe all security-relevant objects, people, and activities "
                "in this surveillance footage. Focus on potential threats. Answer:"
            ) if surveillance_mode else None
            
            inputs = self.processor(
                image,
                text=prompt,
                return_tensors="pt"
            ).to(self.device, torch.float16)
            
            # Generate with configured parameters
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["blip2"]["max_new_tokens"],
                num_beams=self.config["blip2"]["num_beams"],
                early_stopping=True
            )
            
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            if surveillance_mode:
                caption = self._apply_security_tags(caption)
            
            return caption
        
        except Exception as e:
            print(f"Captioning failed for {image_path}: {str(e)}")
            return ""

    def _apply_security_tags(self, caption: str) -> str:
        """
        Apply security tags to suspicious terms in caption.
        
        Args:
            caption: Raw caption string
            
        Returns:
            Tagged caption string
        """
        for term, tag in self.config["security_tags"].items():
            if term.lower() in caption.lower():
                caption = caption.replace(term, tag)
        return caption

    def get_config(self) -> Dict:
        """Return the current configuration."""
        return self.config


if __name__ == "__main__":
    # Example usage
    captioner = BLIP2Captioner()
    test_image = "test_frame.jpg"
    
    print("Configuration:")
    print(yaml.dump(captioner.get_config()))
    
    print("\nGenerated Caption:")
    print(captioner.generate_caption(test_image))