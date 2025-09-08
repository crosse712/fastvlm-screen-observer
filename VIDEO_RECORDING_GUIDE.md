# Video Recording Guide for FastVLM Screen Observer Demo

This guide provides detailed instructions for creating a professional demo video showcasing the FastVLM Screen Observer application.

## 📹 Recording Setup

### Required Tools

#### macOS
- **Built-in**: QuickTime Player or Screenshot app (Cmd+Shift+5)
- **Professional**: OBS Studio (free) - https://obsproject.com

#### Windows
- **Built-in**: Game Bar (Win+G) or Steps Recorder
- **Professional**: OBS Studio (free) - https://obsproject.com

#### Linux
- **SimpleScreenRecorder**: `sudo apt install simplescreenrecorder`
- **OBS Studio**: https://obsproject.com

### Recommended Settings

| Setting | Value | Reason |
|---------|-------|---------|
| Resolution | 1920x1080 | Standard HD |
| Frame Rate | 30 FPS | Smooth playback |
| Format | MP4 (H.264) | Wide compatibility |
| Audio | Include narration | Explain features |
| Duration | 2-3 minutes | Concise demo |

## 🎬 Demo Script

### Pre-Recording Checklist

```bash
# 1. Clean environment
cd /Users/kmh/fastvlm-screen-observer
rm -rf logs/*.ndjson logs/frames/*

# 2. Start fresh instance
./start.sh

# 3. Wait for model to load
# Check: http://localhost:8000/model/status

# 4. Open browser tabs
# - http://localhost:5174 (main app)
# - http://localhost:8000/docs (API docs)
# - Terminal (showing startup)
```

### Scene-by-Scene Script

#### Scene 1: Introduction (0:00-0:15)
```
VISUAL: Terminal showing ./start.sh command
ACTION: Show startup process
NARRATION: 
"Welcome to FastVLM Screen Observer, a real-time screen monitoring 
and analysis system powered by vision-language AI models. Let me 
show you how it works."
```

#### Scene 2: Application Overview (0:15-0:30)
```
VISUAL: Browser at http://localhost:5174
ACTION: Hover over main sections
NARRATION:
"The application has three main sections: the control panel for 
capture settings, the analysis panel showing AI results, and 
real-time logs at the bottom."
```

#### Scene 3: First Screen Capture (0:30-1:00)
```
VISUAL: Click "Capture Screen" button
ACTION: 
1. Show browser permission dialog
2. Select "Entire Screen"
3. Click "Share"
4. Click "Take Screenshot"
NARRATION:
"To capture your screen, simply click the Capture Screen button. 
The browser will ask for permission - select what you want to share. 
Once sharing is active, click Take Screenshot to analyze."
```

#### Scene 4: Analysis Results (1:00-1:30)
```
VISUAL: Analysis panel with results
ACTION: 
1. Point to summary text
2. Scroll through UI elements
3. Show text snippets
4. Highlight any risk flags
NARRATION:
"The AI model instantly analyzes the screen, providing a summary, 
detecting UI elements like buttons and forms, extracting visible text, 
and identifying potential security risks."
```

#### Scene 5: Auto-Capture Mode (1:30-1:50)
```
VISUAL: Enable auto-capture checkbox
ACTION:
1. Check "Auto Capture"
2. Set interval to 5000ms
3. Show multiple captures happening
NARRATION:
"For continuous monitoring, enable auto-capture mode. Set your 
preferred interval, and the system will automatically capture 
and analyze at regular intervals."
```

#### Scene 6: Model Information (1:50-2:10)
```
VISUAL: Open http://localhost:8000/docs
ACTION:
1. Click on /model/status endpoint
2. Click "Try it out"
3. Execute and show response
NARRATION:
"The system currently uses the BLIP vision-language model, running 
on Apple Silicon. You can check the model status and switch between 
different models through the API."
```

#### Scene 7: Export Feature (2:10-2:30)
```
VISUAL: Back to main app
ACTION:
1. Click "Export Logs"
2. Show download notification
3. Open ZIP file
4. Show NDJSON logs
NARRATION:
"All captured data can be exported for analysis. The export includes 
structured logs in NDJSON format and any captured thumbnails, 
making it easy to review sessions later."
```

#### Scene 8: Conclusion (2:30-2:45)
```
VISUAL: Show app with multiple captures
ACTION: Overview shot of full interface
NARRATION:
"FastVLM Screen Observer provides powerful AI-driven screen analysis 
for monitoring, testing, and security applications. Thank you for watching!"
```

## 🎯 Key Points to Showcase

### Must Show
- [x] Screen capture permission flow
- [x] Real-time analysis results
- [x] Auto-capture functionality
- [x] Model status information
- [x] Export capabilities

### Nice to Have
- [ ] Error recovery (deny permission, then retry)
- [ ] Different screen/window/tab selection
- [ ] Browser compatibility info
- [ ] Multiple model comparison

## 🎤 Narration Tips

1. **Clear and Concise**: Speak clearly, avoid filler words
2. **Explain Actions**: Describe what you're doing and why
3. **Highlight Benefits**: Emphasize practical applications
4. **Professional Tone**: Friendly but informative
5. **Practice First**: Do a dry run before recording

## 🎨 Visual Guidelines

### Screen Preparation
```bash
# Clean desktop - hide personal items
# Close unnecessary apps
# Use default browser theme
# Set screen resolution to 1920x1080
# Increase font sizes if needed for visibility
```

### Mouse Movement
- Move deliberately, not frantically
- Pause on important elements
- Use smooth, predictable motions
- Highlight areas before clicking

### Window Management
- Keep windows organized
- Avoid overlapping important content
- Use full screen when possible
- Close unnecessary tabs

## 📝 Post-Production

### Basic Editing
1. **Trim**: Remove dead space at beginning/end
2. **Cut**: Remove any mistakes or long pauses
3. **Annotate**: Add callouts for important features
4. **Captions**: Add subtitles for accessibility

### Tools for Editing
- **iMovie** (macOS): Free, basic editing
- **DaVinci Resolve**: Free, professional features
- **OpenShot**: Free, cross-platform
- **Adobe Premiere**: Paid, professional

### Export Settings
```
Format: MP4
Codec: H.264
Resolution: 1920x1080
Bitrate: 5-10 Mbps
Audio: AAC, 128 kbps
```

## 🚀 Quick Recording with OBS

### OBS Scene Setup
```
1. Install OBS Studio
2. Create Scene: "FastVLM Demo"
3. Add Sources:
   - Display Capture (main screen)
   - Audio Input (microphone)
   - Browser Source (optional overlay)
4. Settings:
   - Output: 1920x1080, 30fps
   - Recording: MP4, High Quality
   - Audio: 128 kbps
```

### OBS Hotkeys
```
Start Recording: Cmd+Shift+R
Stop Recording: Cmd+Shift+R
Pause: Cmd+Shift+P
```

## 📊 Sample Video Structure

```
00:00-00:05 - Title card
00:05-00:15 - Introduction with terminal
00:15-00:30 - Interface overview
00:30-01:00 - Screen capture demo
01:00-01:30 - Analysis results
01:30-01:50 - Auto-capture mode
01:50-02:10 - API and model info
02:10-02:30 - Export feature
02:30-02:45 - Conclusion
02:45-02:50 - End card
```

## ✅ Final Checklist

Before uploading your video:

- [ ] Duration is 2-3 minutes
- [ ] Audio is clear and synchronized
- [ ] All features are demonstrated
- [ ] No sensitive information visible
- [ ] Resolution is at least 720p
- [ ] File size is under 100MB
- [ ] Includes title and description

## 📤 Sharing Your Video

### Recommended Platforms
1. **YouTube**: Public or unlisted
2. **Vimeo**: Professional presentation
3. **GitHub**: Link in README
4. **Google Drive**: For team sharing

### Video Description Template
```
FastVLM Screen Observer - Demo Video

A real-time screen monitoring and analysis system powered by 
vision-language AI models.

Features demonstrated:
- Browser-based screen capture
- AI-powered analysis using BLIP model
- Real-time UI element detection
- Auto-capture mode
- Data export functionality

GitHub: [your-repo-link]
Documentation: [docs-link]

Timestamps:
0:00 - Introduction
0:30 - Screen Capture
1:00 - Analysis Results
1:30 - Auto-Capture
2:10 - Export Feature

#AI #ComputerVision #ScreenCapture #VLM
```

## 🎭 Troubleshooting Recording Issues

| Issue | Solution |
|-------|----------|
| Lag in recording | Lower resolution or framerate |
| No audio | Check microphone permissions |
| Large file size | Use H.264 compression |
| Black screen | Disable hardware acceleration |
| Permission errors | Run OBS as administrator |

## 📚 Additional Resources

- [OBS Studio Guide](https://obsproject.com/wiki/)
- [Screen Recording Best Practices](https://www.techsmith.com/blog/screen-recording-tips/)
- [Video Compression Guide](https://handbrake.fr/docs/)
- [YouTube Creator Guide](https://creatoracademy.youtube.com/)

---

**Remember**: The goal is to create a clear, professional demonstration that showcases the application's capabilities while being easy to follow. Keep it concise, informative, and engaging!