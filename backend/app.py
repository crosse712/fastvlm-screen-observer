"""
Gradio wrapper for Hugging Face Spaces deployment
"""
import gradio as gr
from app.main import app
from fastapi import FastAPI
import uvicorn

# Create a Gradio interface that wraps the FastAPI app
def create_gradio_app():
    # This allows the FastAPI to run alongside Gradio
    return gr.mount_gradio_app(app, path="/")

# For Hugging Face Spaces
demo = create_gradio_app()

if __name__ == "__main__":
    demo.launch()