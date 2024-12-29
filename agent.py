import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import time
from pathlib import Path
import tempfile
from fast_checker import extract_claims, fact_check_claims
import tensorflow_hub as hub
import tensorflow as tf
from moviepy.editor import VideoFileClip
import cv2
import numpy as np

st.set_page_config(
    page_title="VisionVerify - Multimodal AI Agent with Deepfake Detection",
    page_icon="👁️",
    layout="wide"
)

st.title("VisionVerify - Multimodal AI Agent with Deepfake Detection 👁️")

# Initialize single agent with both capabilities
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Multimodal Analyst",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

# Load the TensorFlow model from TensorFlow Hub
@st.cache_resource
def load_detection_model():
    return hub.load("https://tfhub.dev/google/efficientnet/b0/classification/1")

detection_model = load_detection_model()

# Function to preprocess a frame for the TensorFlow model
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    return np.expand_dims(frame_normalized, axis=0)

# Function to detect if a frame is Fake or Real
def detect_fake_or_real(frame):
    input_tensor = preprocess_frame(frame)
    try:
        # Access the default prediction signature
        prediction_fn = detection_model.signatures["default"]
        predictions = prediction_fn(tf.constant(input_tensor))  # Input as a TensorFlow tensor

        # Access the predictions from the "default" key
        probabilities = predictions["default"].numpy()
        return "Fake" if probabilities[0][0] > 0.5 else "Real"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

agent = initialize_agent()

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    user_prompt = st.text_area(
        "What would you like to know?",
        placeholder="Ask any question related to the video - the AI Agent will analyze it and search the web if needed",
        help="You can ask questions about the video content and get relevant information from the web"
    )

    if st.button("Analyze & Research"):
        if not user_prompt:
            st.warning("Please enter your question.")
        else:
            try:
                with st.spinner("Processing video and researching..."):
                    video_file = upload_file(video_path)
                    while video_file.state.name == "PROCESSING":
                        time.sleep(2)
                        video_file = get_file(video_file.name)

                    prompt = f"""
                    First analyze this video and then answer the following question using both 
                    the video analysis and web research: {user_prompt}
                    
                    Provide a comprehensive response focusing on practical, actionable information.
                    """
                    
                    result = agent.run(prompt, videos=[video_file])
                    
                st.subheader("Result")
                st.markdown(result.content)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                Path(video_path).unlink(missing_ok=True)

    # Fake or Real Button
    if st.button("Fake or Real"):
        with st.spinner("Analyzing for deepfake detection..."):
            try:
                video = VideoFileClip(video_path)
                first_frame = video.get_frame(0)  # Extract the first frame
                frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

                # Run the detection
                result = detect_fake_or_real(frame_rgb)
                st.image(frame_rgb, caption=f"Result: {result}")
            except Exception as e:
                st.error(f"An error occurred during deepfake detection: {str(e)}")
else:
    st.info("Please upload a video to begin analysis.")

st.markdown("""
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """, unsafe_allow_html=True)
