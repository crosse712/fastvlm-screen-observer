import { useState, useCallback, useRef } from 'react'
import './ScreenCapture.css'

const ScreenCapture = ({ onCapture, onError }) => {
  const [isCapturing, setIsCapturing] = useState(false)
  const [permissionState, setPermissionState] = useState('prompt') // 'prompt', 'granted', 'denied'
  const [errorMessage, setErrorMessage] = useState(null)
  const [stream, setStream] = useState(null)
  const videoRef = useRef(null)
  const canvasRef = useRef(null)

  const checkBrowserSupport = () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      return {
        supported: false,
        message: 'Screen capture is not supported in your browser. Please use Chrome, Edge, or Firefox.'
      }
    }
    return { supported: true }
  }

  const handlePermissionError = (error) => {
    console.error('Screen capture error:', error)
    
    let userMessage = ''
    let developerInfo = ''
    
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      userMessage = 'Screen capture permission was denied. Please click "Allow" when prompted to share your screen.'
      developerInfo = 'User denied permission'
      setPermissionState('denied')
    } else if (error.name === 'NotFoundError') {
      userMessage = 'No screen capture sources available. Please make sure you have a display connected.'
      developerInfo = 'No capture sources found'
    } else if (error.name === 'NotReadableError') {
      userMessage = 'Screen capture source is currently in use by another application. Please close other screen recording applications and try again.'
      developerInfo = 'Hardware or OS constraint'
    } else if (error.name === 'OverconstrainedError') {
      userMessage = 'The requested screen capture settings are not supported. Trying with default settings...'
      developerInfo = 'Constraint error'
    } else if (error.name === 'TypeError') {
      userMessage = 'Screen capture API error. Please refresh the page and try again.'
      developerInfo = 'API usage error'
    } else if (error.name === 'AbortError') {
      userMessage = 'Screen capture was cancelled.'
      developerInfo = 'User aborted'
    } else {
      userMessage = `Screen capture failed: ${error.message || 'Unknown error'}`
      developerInfo = error.toString()
    }
    
    setErrorMessage(userMessage)
    
    if (onError) {
      onError({
        userMessage,
        technicalDetails: {
          name: error.name,
          message: error.message,
          info: developerInfo
        }
      })
    }
    
    return userMessage
  }

  const startCapture = useCallback(async () => {
    const support = checkBrowserSupport()
    if (!support.supported) {
      setErrorMessage(support.message)
      if (onError) {
        onError({
          userMessage: support.message,
          technicalDetails: { name: 'BrowserNotSupported' }
        })
      }
      return
    }

    setIsCapturing(true)
    setErrorMessage(null)
    
    try {
      // Configure capture options with fallbacks
      const displayMediaOptions = {
        video: {
          displaySurface: 'browser', // Prefer browser tab
          logicalSurface: true,
          cursor: 'always',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false,
        preferCurrentTab: false,
        selfBrowserSurface: 'exclude',
        surfaceSwitching: 'include',
        systemAudio: 'exclude'
      }

      // Try to get display media with full options
      let mediaStream
      try {
        mediaStream = await navigator.mediaDevices.getDisplayMedia(displayMediaOptions)
      } catch (err) {
        console.warn('Failed with full options, trying minimal options:', err)
        // Fallback to minimal options
        mediaStream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: false
        })
      }

      setStream(mediaStream)
      setPermissionState('granted')
      
      // Set up video element to display the stream
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        await videoRef.current.play()
      }

      // Listen for stream end (user stops sharing)
      mediaStream.getVideoTracks()[0].addEventListener('ended', () => {
        stopCapture()
        setErrorMessage('Screen sharing was stopped.')
      })

      // Capture a frame after a short delay to ensure video is ready
      setTimeout(() => captureFrame(mediaStream), 500)
      
    } catch (error) {
      handlePermissionError(error)
    } finally {
      setIsCapturing(false)
    }
  }, [])

  const captureFrame = useCallback((mediaStream) => {
    if (!videoRef.current || !canvasRef.current) {
      setErrorMessage('Unable to capture frame. Video elements not ready.')
      return
    }

    try {
      const video = videoRef.current
      const canvas = canvasRef.current
      const context = canvas.getContext('2d')
      
      // Set canvas size to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      
      // Draw video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      // Convert to blob
      canvas.toBlob((blob) => {
        if (blob && onCapture) {
          // Convert blob to base64 for sending to backend
          const reader = new FileReader()
          reader.onloadend = () => {
            onCapture({
              dataUrl: reader.result,
              blob: blob,
              width: canvas.width,
              height: canvas.height,
              timestamp: new Date().toISOString()
            })
          }
          reader.readAsDataURL(blob)
        }
      }, 'image/png', 0.9)
      
    } catch (error) {
      console.error('Error capturing frame:', error)
      setErrorMessage('Failed to capture frame from screen.')
    }
  }, [onCapture])

  const stopCapture = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }, [stream])

  const retryCapture = useCallback(() => {
    setErrorMessage(null)
    setPermissionState('prompt')
    startCapture()
  }, [startCapture])

  return (
    <div className="screen-capture-container">
      {errorMessage && (
        <div className="error-banner">
          <div className="error-content">
            <svg className="error-icon" viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
            </svg>
            <span className="error-message">{errorMessage}</span>
          </div>
          {permissionState === 'denied' && (
            <button className="retry-button" onClick={retryCapture}>
              Try Again
            </button>
          )}
        </div>
      )}
      
      <div className="capture-controls">
        <button 
          onClick={startCapture}
          disabled={isCapturing || stream}
          className={`capture-button ${isCapturing ? 'capturing' : ''}`}
        >
          {isCapturing ? (
            <>
              <span className="spinner"></span>
              Requesting Permission...
            </>
          ) : stream ? (
            <>
              <span className="recording-dot"></span>
              Screen Sharing Active
            </>
          ) : (
            <>
              <svg className="capture-icon" viewBox="0 0 24 24" width="20" height="20">
                <path fill="currentColor" d="M21 3H3c-1.11 0-2 .89-2 2v14c0 1.11.89 2 2 2h18c1.11 0 2-.89 2-2V5c0-1.11-.89-2-2-2zm0 16H3V5h18v14z"/>
                <path fill="currentColor" d="M15 11l-4-2v6z"/>
              </svg>
              Capture Screen
            </>
          )}
        </button>
        
        {stream && (
          <button onClick={stopCapture} className="stop-button">
            Stop Sharing
          </button>
        )}
      </div>

      {stream && (
        <button 
          onClick={() => captureFrame(stream)}
          className="snapshot-button"
        >
          Take Screenshot
        </button>
      )}

      {/* Hidden video and canvas elements for capture */}
      <video 
        ref={videoRef}
        style={{ display: 'none' }}
        autoPlay
        playsInline
      />
      <canvas 
        ref={canvasRef}
        style={{ display: 'none' }}
      />
      
      {/* Browser compatibility notice */}
      <div className="compatibility-info">
        <details>
          <summary>Browser Compatibility</summary>
          <ul>
            <li>✅ Chrome 72+</li>
            <li>✅ Edge 79+</li>
            <li>✅ Firefox 66+</li>
            <li>✅ Safari 13+ (macOS)</li>
            <li>❌ Internet Explorer</li>
            <li>⚠️ Mobile browsers have limited support</li>
          </ul>
        </details>
      </div>
    </div>
  )
}

export default ScreenCapture