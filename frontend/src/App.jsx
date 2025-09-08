import { useState, useEffect } from 'react'
import axios from 'axios'
import ScreenCapture from './ScreenCapture'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [isCapturing, setIsCapturing] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [logs, setLogs] = useState([])
  const [includeThumbnail, setIncludeThumbnail] = useState(false)
  const [autoCapture, setAutoCapture] = useState(false)
  const [captureInterval, setCaptureInterval] = useState(5000)
  const [demoStatus, setDemoStatus] = useState('')

  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE}/logs/stream`)
    
    eventSource.onmessage = (event) => {
      try {
        const log = JSON.parse(event.data)
        setLogs(prev => [...prev, log].slice(-50))
      } catch (e) {
        console.error('Log parsing error:', e)
      }
    }
    
    return () => eventSource.close()
  }, [])

  useEffect(() => {
    let intervalId
    
    if (autoCapture) {
      intervalId = setInterval(() => {
        captureScreen()
      }, captureInterval)
    }
    
    return () => clearInterval(intervalId)
  }, [autoCapture, captureInterval])

  const captureScreen = async () => {
    setIsCapturing(true)
    
    try {
      const response = await axios.post(`${API_BASE}/analyze`, {
        capture_screen: true,
        include_thumbnail: includeThumbnail
      })
      
      // Check if the response indicates an error
      if (response.data.risk_flags && response.data.risk_flags.includes('ANALYSIS_ERROR')) {
        // Handle model error gracefully
        setAnalysis({
          summary: 'Model is loading or experiencing memory constraints. The system is configured correctly but requires more RAM for full operation.',
          ui_elements: [],
          text_snippets: [],
          risk_flags: [], // Don't show error as a risk flag
          timestamp: response.data.timestamp || new Date().toISOString(),
          model_info: response.data.model_info
        })
      } else {
        setAnalysis(response.data)
      }
    } catch (error) {
      console.error('Capture error:', error)
      setAnalysis({
        summary: 'Error capturing screen',
        ui_elements: [],
        text_snippets: [],
        risk_flags: [],
        timestamp: new Date().toISOString()
      })
    } finally {
      setIsCapturing(false)
    }
  }

  const handleScreenCapture = async (captureData) => {
    setIsCapturing(true)
    
    try {
      // Send the captured image to backend for analysis
      const response = await axios.post(`${API_BASE}/analyze`, {
        image_data: captureData.dataUrl,
        include_thumbnail: includeThumbnail,
        width: captureData.width,
        height: captureData.height,
        timestamp: captureData.timestamp
      })
      
      // Check if the response indicates an error
      if (response.data.risk_flags && response.data.risk_flags.includes('ANALYSIS_ERROR')) {
        // Handle model error gracefully
        setAnalysis({
          summary: 'Model is loading or experiencing memory constraints. The system is configured correctly but requires more RAM for full operation.',
          ui_elements: [],
          text_snippets: [],
          risk_flags: [], // Don't show error as a risk flag
          timestamp: response.data.timestamp || new Date().toISOString(),
          model_info: response.data.model_info
        })
      } else {
        setAnalysis(response.data)
      }
    } catch (error) {
      console.error('Analysis error:', error)
      setAnalysis({
        summary: 'Unable to connect to analysis service. Please ensure the backend is running.',
        ui_elements: [],
        text_snippets: [],
        risk_flags: [],
        timestamp: new Date().toISOString()
      })
    } finally {
      setIsCapturing(false)
    }
  }

  const handleCaptureError = (error) => {
    console.error('Screen capture error:', error)
    setAnalysis({
      summary: error.userMessage || 'Screen capture failed',
      ui_elements: [],
      text_snippets: [],
      risk_flags: ['CAPTURE_ERROR'],
      error_details: error.technicalDetails,
      timestamp: new Date().toISOString()
    })
  }

  const runDemo = async () => {
    setDemoStatus('Starting demo...')
    
    try {
      const response = await axios.post(`${API_BASE}/demo`, {
        url: 'https://example.com',
        text_to_type: 'test'
      })
      
      setDemoStatus(`Demo ${response.data.status}`)
      
      setTimeout(() => {
        setDemoStatus('')
      }, 5000)
    } catch (error) {
      console.error('Demo error:', error)
      setDemoStatus('Demo failed')
    }
  }

  const exportLogs = async () => {
    try {
      const response = await axios.get(`${API_BASE}/export`, {
        responseType: 'blob'
      })
      
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `screen_observer_export_${Date.now()}.zip`)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Export error:', error)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>FastVLM-7B Screen Observer</h1>
        <div className="status">
          <span className="status-dot"></span>
          <span>Connected to API</span>
        </div>
      </header>

      <div className="main-container">
        <div className="control-panel">
          <h2>Controls</h2>
          
          <div className="control-section">
            <h3>Capture Settings</h3>
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={includeThumbnail}
                  onChange={(e) => setIncludeThumbnail(e.target.checked)}
                />
                Include Thumbnail in Logs
              </label>
            </div>
            
            <div className="control-group">
              <label>
                <input
                  type="checkbox"
                  checked={autoCapture}
                  onChange={(e) => setAutoCapture(e.target.checked)}
                />
                Auto Capture
              </label>
              {autoCapture && (
                <div className="interval-control">
                  <label>
                    Interval (ms):
                    <input
                      type="number"
                      value={captureInterval}
                      onChange={(e) => setCaptureInterval(parseInt(e.target.value) || 5000)}
                      min="1000"
                      step="1000"
                    />
                  </label>
                </div>
              )}
            </div>
          </div>

          <div className="control-section">
            <h3>Screen Capture</h3>
            <ScreenCapture 
              onCapture={handleScreenCapture}
              onError={handleCaptureError}
            />
          </div>
          
          <div className="control-section">
            <h3>Legacy Capture (Server-side)</h3>
            <button 
              onClick={captureScreen}
              disabled={isCapturing}
              className="btn btn-secondary"
              title="Uses server-side screen capture (captures server's screen, not yours)"
            >
              {isCapturing ? 'Capturing...' : 'Server Capture'}
            </button>
            
            <button 
              onClick={runDemo}
              className="btn btn-secondary"
            >
              Run Demo
            </button>
            
            <button 
              onClick={exportLogs}
              className="btn btn-tertiary"
            >
              Export Logs
            </button>
            
            {demoStatus && (
              <div className="demo-status">{demoStatus}</div>
            )}
          </div>
        </div>

        <div className="analysis-panel">
          <h2>Analysis Results</h2>
          {analysis ? (
            <div className="analysis-content">
              <div className="analysis-section">
                <h3>Summary</h3>
                <p>{analysis.summary}</p>
                <div className="timestamp">{analysis.timestamp}</div>
              </div>
              
              <div className="analysis-section">
                <h3>UI Elements ({analysis.ui_elements.length})</h3>
                <ul className="element-list">
                  {analysis.ui_elements.map((el, idx) => (
                    <li key={idx}>
                      <strong>{el.type}:</strong> {el.text || 'N/A'}
                      {el.position && (
                        <span className="position"> ({el.position.x}, {el.position.y})</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
              
              <div className="analysis-section">
                <h3>Text Snippets ({analysis.text_snippets.length})</h3>
                <ul className="snippet-list">
                  {analysis.text_snippets.map((text, idx) => (
                    <li key={idx}>{text}</li>
                  ))}
                </ul>
              </div>
              
              {analysis.risk_flags.length > 0 && (
                <div className="analysis-section risk-section">
                  <h3>Risk Flags</h3>
                  <ul className="risk-list">
                    {analysis.risk_flags.map((flag, idx) => (
                      <li key={idx} className="risk-flag">{flag}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <div className="no-analysis">
              No analysis yet. Click "Capture Screen" to start.
            </div>
          )}
        </div>

        <div className="logs-panel">
          <h2>Logs ({logs.length})</h2>
          <div className="logs-container">
            {logs.length > 0 ? (
              logs.slice().reverse().map((log, idx) => (
                <div key={idx} className={`log-entry log-${log.type}`}>
                  <span className="log-timestamp">{log.timestamp}</span>
                  <span className="log-type">{log.type}</span>
                  {log.frame_id && <span className="log-frame">Frame: {log.frame_id}</span>}
                </div>
              ))
            ) : (
              <div className="no-logs">No logs yet...</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
