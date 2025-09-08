#!/usr/bin/env python3
"""
Generate sample logs for FastVLM Screen Observer
This script creates realistic NDJSON logs with various analysis results
"""

import json
import requests
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

API_BASE = "http://localhost:8000"
LOGS_DIR = "logs"
SAMPLE_LOGS_FILE = "logs/sample_logs.ndjson"

def ensure_directories():
    """Ensure logs directory exists"""
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(f"{LOGS_DIR}/frames", exist_ok=True)

def create_test_image(scenario="default"):
    """Create different test images for various scenarios"""
    
    if scenario == "login":
        # Create login screen
        img = Image.new('RGB', (1920, 1080), color='#f0f0f0')
        draw = ImageDraw.Draw(img)
        
        # Draw login form
        draw.rectangle([660, 340, 1260, 740], fill='white', outline='#ddd')
        draw.text((880, 380), "Login to System", fill='#333')
        
        # Username field
        draw.rectangle([760, 460, 1160, 510], fill='white', outline='#999')
        draw.text((770, 475), "Username", fill='#666')
        
        # Password field
        draw.rectangle([760, 530, 1160, 580], fill='white', outline='#999')
        draw.text((770, 545), "••••••••", fill='#666')
        
        # Login button
        draw.rectangle([760, 620, 1160, 680], fill='#2196F3', outline='#1976D2')
        draw.text((920, 640), "Sign In", fill='white')
        
        description = "Login form with username and password fields"
        
    elif scenario == "dashboard":
        # Create dashboard screen
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        # Header
        draw.rectangle([0, 0, 1920, 80], fill='#333')
        draw.text((50, 30), "Analytics Dashboard", fill='white')
        
        # Stats cards
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336']
        titles = ['Users', 'Revenue', 'Orders', 'Alerts']
        values = ['1,234', '$45,678', '89', '3']
        
        for i, (color, title, value) in enumerate(zip(colors, titles, values)):
            x = 100 + i * 450
            draw.rectangle([x, 150, x+400, 300], fill=color)
            draw.text((x+20, 170), title, fill='white')
            draw.text((x+20, 220), value, fill='white')
        
        # Chart area
        draw.rectangle([100, 350, 900, 750], fill='#fafafa', outline='#ddd')
        draw.text((450, 540), "Chart Area", fill='#999')
        
        # Table
        draw.rectangle([1000, 350, 1820, 750], fill='#fafafa', outline='#ddd')
        draw.text((1350, 380), "Recent Activity", fill='#333')
        
        description = "Analytics dashboard with charts and statistics"
        
    elif scenario == "code_editor":
        # Create code editor screen
        img = Image.new('RGB', (1920, 1080), color='#1e1e1e')
        draw = ImageDraw.Draw(img)
        
        # Editor tabs
        draw.rectangle([0, 0, 1920, 40], fill='#2d2d2d')
        draw.text((20, 12), "main.py", fill='white')
        draw.text((120, 12), "utils.py", fill='#888')
        
        # Line numbers
        for i in range(1, 30):
            draw.text((20, 50 + i*25), str(i), fill='#666')
        
        # Code content
        code_lines = [
            "def process_data(input_file):",
            "    '''Process input data file'''",
            "    with open(input_file, 'r') as f:",
            "        data = json.load(f)",
            "    ",
            "    results = []",
            "    for item in data:",
            "        processed = transform(item)",
            "        results.append(processed)",
            "    ",
            "    return results",
            "",
            "def transform(item):",
            "    '''Transform single data item'''",
            "    return {",
            "        'id': item.get('id'),",
            "        'value': item.get('value') * 2,",
            "        'timestamp': datetime.now()",
            "    }"
        ]
        
        for i, line in enumerate(code_lines):
            draw.text((70, 75 + i*25), line, fill='#d4d4d4')
        
        # Sidebar
        draw.rectangle([1700, 40, 1920, 1080], fill='#252525')
        draw.text((1720, 60), "Explorer", fill='white')
        
        description = "Code editor showing Python script"
        
    elif scenario == "sensitive":
        # Create screen with sensitive data
        img = Image.new('RGB', (1920, 1080), color='white')
        draw = ImageDraw.Draw(img)
        
        # Warning banner
        draw.rectangle([0, 0, 1920, 60], fill='#FFF3CD')
        draw.text((50, 20), "⚠️ Sensitive Information - Handle with Care", fill='#856404')
        
        # Credit card info (masked)
        draw.rectangle([100, 150, 700, 350], fill='#f8f9fa', outline='#dc3545')
        draw.text((120, 170), "Payment Information", fill='#dc3545')
        draw.text((120, 220), "Card Number: **** **** **** 1234", fill='#333')
        draw.text((120, 260), "CVV: ***", fill='#333')
        draw.text((120, 300), "Expiry: 12/25", fill='#333')
        
        # Personal info
        draw.rectangle([800, 150, 1400, 350], fill='#f8f9fa', outline='#dc3545')
        draw.text((820, 170), "Personal Details", fill='#dc3545')
        draw.text((820, 220), "SSN: ***-**-6789", fill='#333')
        draw.text((820, 260), "DOB: 01/15/1990", fill='#333')
        
        # API Keys
        draw.rectangle([100, 450, 1400, 600], fill='#fff5f5', outline='#dc3545')
        draw.text((120, 470), "API Configuration", fill='#dc3545')
        draw.text((120, 520), "API_KEY=sk-...REDACTED", fill='#666')
        draw.text((120, 560), "SECRET=sec_...REDACTED", fill='#666')
        
        description = "Screen containing sensitive financial and personal information"
        
    else:  # default
        # Create generic application screen
        img = Image.new('RGB', (1280, 720), color='white')
        draw = ImageDraw.Draw(img)
        
        # Header
        draw.rectangle([0, 0, 1280, 60], fill='#4a90e2')
        draw.text((20, 20), "Application Window", fill='white')
        
        # Buttons
        draw.rectangle([100, 100, 250, 150], fill='#5cb85c')
        draw.text((150, 115), "Save", fill='white')
        
        draw.rectangle([300, 100, 450, 150], fill='#f0ad4e')
        draw.text((340, 115), "Cancel", fill='white')
        
        # Text area
        draw.rectangle([100, 200, 1180, 500], fill='#f5f5f5', outline='#ddd')
        draw.text((120, 220), "Sample text content here", fill='#333')
        
        description = "Generic application window with buttons"
    
    return img, description

def generate_sample_logs():
    """Generate various sample log entries"""
    
    print("Generating sample logs...")
    ensure_directories()
    
    scenarios = [
        ("default", "Generic application"),
        ("login", "Login screen"),
        ("dashboard", "Analytics dashboard"),
        ("code_editor", "Code editor"),
        ("sensitive", "Sensitive data screen")
    ]
    
    logs = []
    
    # Check API status first
    try:
        response = requests.get(f"{API_BASE}/model/status")
        model_status = response.json()
        print(f"Model Status: {model_status['model_type']} on {model_status['device']}")
    except Exception as e:
        print(f"Warning: API not responding: {e}")
        print("Generating mock logs instead...")
        model_status = {"model_type": "mock", "device": "cpu"}
    
    # Generate logs for each scenario
    for scenario_type, scenario_name in scenarios:
        print(f"\nProcessing scenario: {scenario_name}")
        
        # Create test image
        img, description = create_test_image(scenario_type)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Generate frame ID and timestamp
        frame_id = f"frame_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()
        
        # Log frame capture
        logs.append({
            "timestamp": timestamp,
            "type": "frame_capture",
            "frame_id": frame_id,
            "scenario": scenario_name,
            "has_thumbnail": True
        })
        
        # Try to analyze with API
        try:
            response = requests.post(
                f"{API_BASE}/analyze",
                json={
                    "image_data": f"data:image/png;base64,{img_base64}",
                    "include_thumbnail": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_log = {
                    "timestamp": datetime.now().isoformat(),
                    "type": "analysis",
                    "frame_id": frame_id,
                    "scenario": scenario_name,
                    "summary": result.get("summary", description),
                    "ui_elements": result.get("ui_elements", []),
                    "text_snippets": result.get("text_snippets", []),
                    "risk_flags": result.get("risk_flags", [])
                }
            else:
                raise Exception(f"API returned {response.status_code}")
                
        except Exception as e:
            print(f"  API analysis failed: {e}, using mock data")
            # Generate mock analysis
            analysis_log = generate_mock_analysis(scenario_type, frame_id, description)
        
        logs.append(analysis_log)
        
        # Add some automation logs for certain scenarios
        if scenario_type in ["login", "dashboard"]:
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "type": "automation",
                "frame_id": frame_id,
                "action": "click" if scenario_type == "login" else "scroll",
                "target": "button#submit" if scenario_type == "login" else "div.chart-container",
                "success": True
            })
        
        # Small delay between scenarios
        time.sleep(0.5)
    
    # Write logs to file
    with open(SAMPLE_LOGS_FILE, 'w') as f:
        for log in logs:
            f.write(json.dumps(log) + '\n')
    
    print(f"\n✅ Sample logs generated: {SAMPLE_LOGS_FILE}")
    print(f"   Total entries: {len(logs)}")
    
    # Also create a pretty-printed version for review
    pretty_file = SAMPLE_LOGS_FILE.replace('.ndjson', '_pretty.json')
    with open(pretty_file, 'w') as f:
        json.dump(logs, f, indent=2)
    print(f"   Pretty version: {pretty_file}")
    
    return logs

def generate_mock_analysis(scenario_type, frame_id, description):
    """Generate mock analysis data for when API is not available"""
    
    mock_data = {
        "default": {
            "ui_elements": [
                {"type": "button", "text": "Save", "position": {"x": 150, "y": 115}},
                {"type": "button", "text": "Cancel", "position": {"x": 340, "y": 115}},
                {"type": "textarea", "text": "Text input area", "position": {"x": 640, "y": 350}}
            ],
            "text_snippets": ["Application Window", "Save", "Cancel", "Sample text content here"],
            "risk_flags": []
        },
        "login": {
            "ui_elements": [
                {"type": "input", "text": "Username field", "position": {"x": 960, "y": 485}},
                {"type": "input", "text": "Password field", "position": {"x": 960, "y": 555}},
                {"type": "button", "text": "Sign In", "position": {"x": 960, "y": 650}}
            ],
            "text_snippets": ["Login to System", "Username", "Sign In"],
            "risk_flags": ["AUTH_FORM", "PASSWORD_FIELD"]
        },
        "dashboard": {
            "ui_elements": [
                {"type": "card", "text": "Users: 1,234", "position": {"x": 300, "y": 225}},
                {"type": "card", "text": "Revenue: $45,678", "position": {"x": 750, "y": 225}},
                {"type": "chart", "text": "Chart Area", "position": {"x": 500, "y": 550}},
                {"type": "table", "text": "Recent Activity", "position": {"x": 1410, "y": 550}}
            ],
            "text_snippets": ["Analytics Dashboard", "Users", "Revenue", "Orders", "Alerts"],
            "risk_flags": []
        },
        "code_editor": {
            "ui_elements": [
                {"type": "tab", "text": "main.py", "position": {"x": 60, "y": 20}},
                {"type": "editor", "text": "Code editor", "position": {"x": 960, "y": 540}},
                {"type": "sidebar", "text": "Explorer", "position": {"x": 1810, "y": 560}}
            ],
            "text_snippets": ["def process_data", "json.load", "transform", "return results"],
            "risk_flags": ["SOURCE_CODE"]
        },
        "sensitive": {
            "ui_elements": [
                {"type": "warning", "text": "Sensitive Information", "position": {"x": 960, "y": 30}},
                {"type": "form", "text": "Payment Information", "position": {"x": 400, "y": 250}},
                {"type": "form", "text": "Personal Details", "position": {"x": 1100, "y": 250}}
            ],
            "text_snippets": ["Card Number: ****", "SSN: ***", "API_KEY=", "SECRET="],
            "risk_flags": ["SENSITIVE_DATA", "CREDIT_CARD", "PII", "API_KEYS", "HIGH_RISK"]
        }
    }
    
    data = mock_data.get(scenario_type, mock_data["default"])
    
    return {
        "timestamp": datetime.now().isoformat(),
        "type": "analysis",
        "frame_id": frame_id,
        "scenario": scenario_type,
        "summary": f"[MOCK] {description}",
        "ui_elements": data["ui_elements"],
        "text_snippets": data["text_snippets"],
        "risk_flags": data["risk_flags"],
        "mock_mode": True
    }

if __name__ == "__main__":
    try:
        generate_sample_logs()
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()