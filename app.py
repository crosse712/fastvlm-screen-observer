"""
Simple test app for Hugging Face Spaces - will be replaced by full app
"""
import gradio as gr
import subprocess
import os
import time

def get_status():
    """Check if services are running"""
    try:
        # Check if nginx is running
        nginx_status = "✅ Nginx is configured" if os.path.exists('/etc/nginx/sites-enabled/default') else "❌ Nginx not configured"
        
        # Check if backend directory exists
        backend_status = "✅ Backend code present" if os.path.exists('/app/backend/app/main.py') else "❌ Backend code missing"
        
        # Check if frontend was built
        frontend_status = "✅ Frontend built" if os.path.exists('/usr/share/nginx/html/index.html') else "❌ Frontend not built"
        
        return f"""
        # FastVLM Screen Observer Status
        
        {nginx_status}
        {backend_status}
        {frontend_status}
        
        The full application is being deployed. Please check back in a few moments.
        
        Visit: http://localhost:7860 once services are running.
        """
    except Exception as e:
        return f"Error checking status: {str(e)}"

# Create simple Gradio interface for testing
demo = gr.Interface(
    fn=get_status,
    inputs=None,
    outputs="markdown",
    title="FastVLM Screen Observer - Deployment Status",
    description="Checking deployment status..."
)

if __name__ == "__main__":
    # Try to start supervisor in background
    try:
        subprocess.Popen(["/app/start.sh"])
        time.sleep(2)
    except:
        pass
    
    # Launch Gradio on port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)