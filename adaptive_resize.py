import torch
import torch.nn.functional as F
import math

class AdaptiveImageResize:
    """
    ComfyUI node for adaptive image resizing based on target pixel counts.
    Maintains aspect ratio while hitting specific pixel count targets.
    """
    
    def __init__(self):
        # Define target pixel counts based on model and quality
        self.target_pixels = {
            "720p": {
                "high": 921600,    # 720p max (720² * 16/9)
                "medium": 691200,  # ~75% of max
                "low": 460800      # ~50% of max
            },
            "480p": {
                "high": 409600,    # 480p max (480² * 16/9)
                "medium": 307200,  # ~75% of max
                "low": 204800      # ~50% of max
            }
        }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["720p", "480p"], {"default": "720p"}),
                "quality": (["high", "medium", "low"], {"default": "high"}),
                "fast_mode": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height", "total_pixels")
    FUNCTION = "resize_image"
    CATEGORY = "image/transform"
    
    def resize_image(self, image, model, quality, fast_mode=True):
        """
        Resize image to target pixel count while maintaining aspect ratio.
        
        Args:
            image: Input image tensor (batch, height, width, channels)
            model: Target resolution model ("720p" or "480p")
            quality: Quality setting ("high", "medium", "low")
            
        Returns:
            Resized image tensor, new width, new height, total pixels
        """
        # Get target pixel count
        target_pixel_count = self.target_pixels[model][quality]
        
        # Get current image dimensions
        batch_size, current_height, current_width, channels = image.shape
        
        # Calculate aspect ratio
        aspect_ratio = current_width / current_height
        
        # Calculate dimensions that maintain aspect ratio and hit target pixel count
        # total_pixels = sqrt(target_pixel_count * aspect_ratio)
        total_pixels = math.sqrt(target_pixel_count * aspect_ratio)
        new_width = round(total_pixels)
        new_height = round(total_pixels / aspect_ratio)
        
        # Ensure dimensions are multiples of 16 (common requirement for video/AI processing)
        new_width = round(new_width / 16) * 16
        new_height = round(new_height / 16) * 16
        
        # Ensure minimum dimensions
        new_width = max(new_width, 16)
        new_height = max(new_height, 16)
        
        # Convert to int
        new_width = int(new_width)
        new_height = int(new_height)
        
        # Only resize if dimensions actually changed
        if new_width == current_width and new_height == current_height:
            return (image, new_width, new_height, new_width * new_height)
        
        # Resize the image with optimizations
        # ComfyUI images are in format (batch, height, width, channels)
        # PyTorch expects (batch, channels, height, width) for interpolation
        
        # Check if we're downscaling significantly (>2x) - use area interpolation for speed
        scale_factor = (current_width * current_height) / (new_width * new_height)
        
        if fast_mode:
            if scale_factor > 4.0:
                # For large downscales, use area interpolation (fastest)
                interpolation_mode = 'area'
            else:
                # For small changes or upscaling, use bilinear (faster than bicubic)
                interpolation_mode = 'bilinear'
        else:
            # Quality mode - use bicubic for better results
            interpolation_mode = 'bicubic'
        
        # Use in-place operations and avoid unnecessary memory allocation
        with torch.no_grad():  # Disable gradient computation for speed
            image_resized = image.permute(0, 3, 1, 2)  # (batch, channels, height, width)
            
            # Resize with optimized interpolation
            image_resized = F.interpolate(
                image_resized, 
                size=(new_height, new_width), 
                mode=interpolation_mode, 
                align_corners=False,
                antialias=True if interpolation_mode == 'bilinear' else False
            )
            
            # Convert back to ComfyUI format
            image_resized = image_resized.permute(0, 2, 3, 1)  # (batch, height, width, channels)
            
            # Clamp values to [0, 1] range (in-place for speed)
            image_resized.clamp_(0.0, 1.0)
        
        return (image_resized, new_width, new_height, new_width * new_height)

# Node mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "AdaptiveImageResize": AdaptiveImageResize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveImageResize": "Adaptive Image Resize"
}