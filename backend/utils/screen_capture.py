import mss
import mss.tools
from PIL import Image
import io
import numpy as np
from typing import Optional

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
    
    def capture(self, monitor_index: int = 0) -> bytes:
        try:
            if monitor_index == 0:
                monitor = self.sct.monitors[0]
            else:
                monitor = self.sct.monitors[monitor_index]
            
            screenshot = self.sct.grab(monitor)
            
            img = Image.frombytes(
                "RGB",
                (screenshot.width, screenshot.height),
                screenshot.rgb
            )
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            return img_byte_arr.getvalue()
            
        except Exception as e:
            print(f"Screen capture error: {e}")
            return self._create_placeholder_image()
    
    def create_thumbnail(self, image_data: bytes, size: tuple = (320, 240)) -> bytes:
        try:
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(size, Image.Resampling.LANCZOS)
            
            thumb_byte_arr = io.BytesIO()
            img.save(thumb_byte_arr, format='PNG')
            thumb_byte_arr.seek(0)
            
            return thumb_byte_arr.getvalue()
            
        except Exception as e:
            print(f"Thumbnail creation error: {e}")
            return image_data
    
    def _create_placeholder_image(self) -> bytes:
        img = Image.new('RGB', (1920, 1080), color='gray')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        return img_byte_arr.getvalue()