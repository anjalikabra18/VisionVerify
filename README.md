# VisionVerify: Multimodal AI Agent with Deepfake Detection 👁️

**VisionVerify** is a next-generation AI application that combines video analysis and web research into one seamless experience. By leveraging the power of Google's Gemini 2.0 Flash model, TensorFlow, and the Phidata framework, VisionVerify delivers advanced capabilities for video understanding, deepfake detection, and real-time information retrieval.

---

## Features

**Analyze & Research**  
- Upload videos and ask questions about their content.  
- Get detailed answers by combining video analysis with real-time web research powered by Gemini 2.0 Flash and DuckDuckGo.

**Fake or Real Detection**  
- Detect video authenticity with TensorFlow’s EfficientNet model.  
- Analyze frames to determine if videos are manipulated or genuine.

**Video File Support**  
- Compatible with common formats like MP4, MOV, and AVI.

**User-Friendly Interface**  
- Interactive design built with Streamlit for a smooth and intuitive experience.  
- Real-time video processing and results display.

---

## About VisionVerify

VisionVerify combines video content analysis with web-based research to create an unparalleled AI tool.  
- **Enhanced Speed and Performance:** Gemini 2.0 Flash offers 2x faster processing with features like image generation, speech synthesis, and built-in tool integration.  
- **Free Experimental API:** The API is accessible with generous limits during its experimental phase.  
- **Deepfake Detection:** Powered by TensorFlow, ensuring reliable results in analyzing video authenticity.  
- **Phidata Framework:** Simplifies development with robust integration for multimodal agents and external tools.

---

## Installation

### Prerequisites  
- Python 3.8+  
- Virtual Environment (recommended)  
- Git (to clone the repository)  

## Usage

1. **Upload Video**  
   Select your video file using the "Upload a video file" button.  

2. **Analyze & Research**  
   Enter your question about the video’s content in the text area.  
   Click "Analyze & Research" to get insights combining video analysis and web search.  

3. **Fake or Real Detection**  
   Click "Fake or Real" to determine whether the video is authentic or manipulated.

---

## Project Structure

```
VisionVerify/
├── .env                    Environment variables (optional)
├── agent.py                Multimodal AI agent integration
├── fast_checker.py         Fact-checking and claim verification
├── gem_env/                Virtual environment (ignored)
├── requirements.txt        Python dependencies
├── multimodal_agent.py     Main application file
└── README.md               Project documentation
```

---

## Technologies Used

**Phidata Framework**  
Streamlines multimodal AI agent development, enabling integration of Gemini 2.0 Flash and external tools.  

**Streamlit**  
Provides an interactive, user-friendly interface.  

**TensorFlow**  
Powering deepfake detection with pretrained models.  

**TensorFlow Hub**  
EfficientNet for image classification and video analysis.  

**DuckDuckGo API**  
For real-time web research and question answering.  

---

## Models

**EfficientNet**  
Pretrained TensorFlow model for image classification, adapted for detecting video manipulation.  

**Gemini 2.0 Flash**  
Google’s advanced multimodal model for video analysis and integrated web research.  

---

## Contributing

We welcome contributions!  
1. Fork the repository.  
2. Create a new branch for your feature or bug fix.  
3. Commit your changes and submit a pull request.  

---

## License

This project is licensed under the [MIT License](LICENSE).  

---

## Contact

**Anjali Kabra**  
[GitHub Profile](https://github.com/anjalikabra18)  
