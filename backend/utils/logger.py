import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import base64

class NDJSONLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.frames_dir = self.log_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "logs.ndjson"
    
    def log_frame(self, frame_id: str, thumbnail: Optional[bytes], timestamp: str):
        try:
            if thumbnail:
                frame_path = self.frames_dir / f"{frame_id}.png"
                with open(frame_path, "wb") as f:
                    f.write(thumbnail)
                
                thumbnail_b64 = base64.b64encode(thumbnail).decode('utf-8')
            else:
                thumbnail_b64 = None
            
            log_entry = {
                "type": "frame_capture",
                "timestamp": timestamp,
                "frame_id": frame_id,
                "thumbnail": thumbnail_b64 if thumbnail_b64 else None,
                "has_thumbnail": thumbnail is not None
            }
            
            self._write_log(log_entry)
            
        except Exception as e:
            print(f"Frame logging error: {e}")
    
    def log_analysis(self, analysis_data: Dict[str, Any]):
        try:
            log_entry = {
                "type": "analysis",
                "timestamp": analysis_data.get("timestamp", datetime.now().isoformat()),
                "data": analysis_data
            }
            
            self._write_log(log_entry)
            
        except Exception as e:
            print(f"Analysis logging error: {e}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        try:
            log_entry = {
                "type": event_type,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            self._write_log(log_entry)
            
        except Exception as e:
            print(f"Event logging error: {e}")
    
    def _write_log(self, entry: Dict[str, Any]):
        try:
            with open(self.log_file, "a") as f:
                json.dump(entry, f)
                f.write("\n")
                f.flush()
                
        except Exception as e:
            print(f"Write log error: {e}")
    
    def clear_logs(self):
        try:
            if self.log_file.exists():
                self.log_file.unlink()
            
            for frame_file in self.frames_dir.glob("*.png"):
                frame_file.unlink()
                
        except Exception as e:
            print(f"Clear logs error: {e}")