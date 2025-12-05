import streamlit as st
import cv2
import numpy as np
import requests
import time
from collections import Counter

# Page config
st.set_page_config(
    page_title="♻️ Waste Detection Live", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("♻️ Live Waste Detection System")
st.markdown("Real-time construction waste detection using computer vision")

# API Configuration
API_URL = "http://localhost:8000/detect"
HEALTH_URL = "http://localhost:8000/health"

# Class definitions
CLASS_NAMES = [
    'brick', 'concrete', 'foam', 'general_w', 'gypsum_board',
    'pipes', 'plastic', 'stone', 'tile', 'wood'
]

CLASS_COLORS = {
    'brick': (255, 0, 0),
    'concrete': (128, 128, 128),
    'foam': (255, 255, 0),
    'general_w': (0, 255, 255),
    'gypsum_board': (255, 165, 0),
    'pipes': (0, 0, 255),
    'plastic': (0, 255, 0),
    'stone': (160, 82, 45),
    'tile': (255, 0, 255),
    'wood': (139, 69, 19)
}

# Initialize session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'total_detections' not in st.session_state:
    st.session_state.total_detections = 0
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'available_cameras' not in st.session_state:
    st.session_state.available_cameras = None
if 'camera_detection_done' not in st.session_state:
    st.session_state.camera_detection_done = False


def detect_cameras():
    """Detect all available cameras and their working backends."""
    st.info("🔍 Scanning for available cameras...")
    
    available = []
    backends = [
        (cv2.CAP_V4L2, "V4L2 (Linux)"),
        (cv2.CAP_ANY, "Auto"),
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_AVFOUNDATION, "AVFoundation (macOS)")
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Check cameras 0-5
    for cam_idx in range(6):
        progress_bar.progress((cam_idx + 1) / 6)
        status_text.text(f"Testing camera {cam_idx}...")
        
        for backend, backend_name in backends:
            try:
                cap = cv2.VideoCapture(cam_idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        available.append({
                            'index': cam_idx,
                            'backend': backend,
                            'backend_name': backend_name,
                            'resolution': f"{w}x{h}",
                            'width': w,
                            'height': h
                        })
                        cap.release()
                        break  # Found working backend for this camera
                cap.release()
            except:
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    return available


def draw_boxes(frame, boxes, scores, classes):
    """Draw bounding boxes on frame."""
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        c = int(c)
        label = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"cls{c}"
        color = CLASS_COLORS.get(label, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label background
        label_text = f"{label} {s:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, 
            (int(x1), int(y1) - text_height - 10),
            (int(x1) + text_width, int(y1)),
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame, label_text, 
            (int(x1), int(y1) - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    
    return frame


def process_detections(classes):
    """Process detection statistics."""
    if classes:
        class_labels = [CLASS_NAMES[int(c)] for c in classes if int(c) < len(CLASS_NAMES)]
        return Counter(class_labels)
    return Counter()


# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Check API health
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        if response.status_code == 200:
            data = response.json()
            if data.get('model_loaded'):
                st.success("✅ API Connected & Model Loaded")
            else:
                st.warning("⚠️ API Connected but Model Not Loaded")
        else:
            st.error("❌ API Unhealthy")
    except:
        st.error("❌ API Disconnected - Start api.py first!")
    
    st.markdown("---")
    
    # Camera Detection
    st.subheader("📹 Camera Setup")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Detect Cameras", use_container_width=True):
            st.session_state.available_cameras = detect_cameras()
            st.session_state.camera_detection_done = True
            st.rerun()
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.available_cameras = None
            st.session_state.camera_detection_done = False
            st.rerun()
    
    # Show available cameras
    if st.session_state.available_cameras is not None:
        if len(st.session_state.available_cameras) == 0:
            st.error("❌ No cameras found!")
            st.info("💡 Make sure your camera is connected and not in use by another app")
        else:
            st.success(f"✅ Found {len(st.session_state.available_cameras)} camera(s)")
            
            # Create camera options
            camera_options = []
            for cam in st.session_state.available_cameras:
                label = f"Camera {cam['index']} ({cam['backend_name']}) - {cam['resolution']}"
                camera_options.append(label)
            
            selected_camera_label = st.selectbox(
                "Select Camera",
                camera_options,
                help="Choose the camera to use"
            )
            
            # Get selected camera config
            selected_idx = camera_options.index(selected_camera_label)
            selected_camera = st.session_state.available_cameras[selected_idx]
            
            # Display camera info
            with st.expander("📊 Camera Details"):
                st.write(f"**Index:** {selected_camera['index']}")
                st.write(f"**Backend:** {selected_camera['backend_name']}")
                st.write(f"**Native Resolution:** {selected_camera['resolution']}")
    else:
        st.info("👆 Click 'Detect Cameras' to scan for available cameras")
        # Manual fallback
        st.markdown("**Or manually configure:**")
        selected_camera = {
            'index': st.number_input("Camera Index", 0, 10, 0),
            'backend': cv2.CAP_ANY,
            'backend_name': "Auto"
        }
    
    st.markdown("---")
    
    # Resolution settings
    st.subheader("🎬 Video Settings")
    resolution_presets = {
        "480p (640x480)": (640, 480),
        "720p (1280x720)": (1280, 720),
        "1080p (1920x1080)": (1920, 1080),
        "Custom": None
    }
    
    resolution_choice = st.selectbox(
        "Resolution",
        list(resolution_presets.keys()),
        index=1
    )
    
    if resolution_choice == "Custom":
        resolution_width = st.number_input("Width", 320, 3840, 1280)
        resolution_height = st.number_input("Height", 240, 2160, 720)
    else:
        resolution_width, resolution_height = resolution_presets[resolution_choice]
    
    # Detection settings
    st.subheader("🎯 Detection Settings")
    conf_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.25, 0.05,
        help="Minimum confidence for detections"
    )
    
    # Display settings
    st.subheader("🖼️ Display Settings")
    show_fps = st.checkbox("Show FPS", value=True)
    show_count = st.checkbox("Show Detection Count", value=True)
    frame_skip = st.slider("Frame Skip", 0, 5, 0, help="Skip frames for better performance")
    jpeg_quality = st.slider("JPEG Quality", 50, 100, 85, help="Lower = faster, higher = better quality")
    
    # Statistics
    st.subheader("📊 Session Statistics")
    st.metric("Total Detections", st.session_state.total_detections)
    
    if st.button("🔄 Reset Statistics"):
        st.session_state.total_detections = 0
        st.session_state.detection_history = []
        st.rerun()


# Main UI
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 Live Feed")
    
    # Control buttons
    button_col1, button_col2, button_col3 = st.columns(3)
    with button_col1:
        start_disabled = st.session_state.is_running or (st.session_state.available_cameras is not None and len(st.session_state.available_cameras) == 0)
        if st.button("▶️ Start", use_container_width=True, disabled=start_disabled):
            st.session_state.is_running = True
            st.rerun()
    
    with button_col2:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not st.session_state.is_running):
            st.session_state.is_running = False
            st.rerun()
    
    with button_col3:
        save_frame = st.button("📸 Save Frame", use_container_width=True)
    
    # Video frame placeholder
    FRAME_WINDOW = st.empty()
    
    # Status placeholders
    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        fps_text = st.empty()
    with status_col2:
        detection_text = st.empty()
    with status_col3:
        camera_status = st.empty()

with col2:
    st.subheader("🔍 Current Detections")
    detection_stats = st.empty()

# Session management
session = requests.Session()

# Main processing loop
if st.session_state.is_running:
    cap = None
    frame_count = 0
    
    # Get camera config
    if st.session_state.available_cameras and len(st.session_state.available_cameras) > 0:
        cam_config = selected_camera
    else:
        # Fallback to manual config
        cam_config = selected_camera
    
    try:
        # Open camera with detected backend
        cap = cv2.VideoCapture(cam_config['index'], cam_config['backend'])
        
        if not cap.isOpened():
            st.error(f"❌ Cannot open camera {cam_config['index']} with {cam_config['backend_name']}")
            st.info("💡 Try clicking 'Detect Cameras' to find available cameras")
            st.session_state.is_running = False
        else:
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution_height)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Get actual resolution (might differ from requested)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            camera_status.success(f"📹 {actual_width}x{actual_height}")
            
            while st.session_state.is_running:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Cannot read from camera")
                    break
                
                frame_count += 1
                
                # Skip frames if configured
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue
                
                # Encode frame with configurable quality
                _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                files = {"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
                
                # Send to API with retry
                boxes, scores, classes = [], [], []
                try:
                    response = session.post(
                        API_URL, 
                        files=files, 
                        params={"conf_threshold": conf_threshold},
                        timeout=5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        boxes = data.get("boxes", [])
                        scores = data.get("scores", [])
                        classes = data.get("classes", [])
                        
                        if boxes:
                            frame = draw_boxes(frame, boxes, scores, classes)
                            st.session_state.total_detections += len(boxes)
                    else:
                        detection_text.warning(f"⚠️ API Error: {response.status_code}")
                
                except requests.exceptions.Timeout:
                    detection_text.warning("⚠️ API Timeout")
                except requests.exceptions.RequestException as e:
                    detection_text.error(f"⚠️ Connection Error")
                
                # Save frame if requested
                if save_frame and frame is not None:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    st.success(f"✅ Saved as {filename}")
                
                # Calculate and display FPS
                fps = 1.0 / (time.time() - start_time)
                
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, width='content')
                
                # Update status
                if show_fps:
                    fps_text.markdown(f"**⚡ FPS:** {fps:.1f}")
                
                if show_count:
                    detection_text.markdown(f"**🎯 Detections:** {len(boxes)}")
                
                # Update detection statistics
                if boxes:
                    stats = process_detections(classes)
                    stats_text = "### Current Frame\n"
                    for label, count in stats.most_common():
                        stats_text += f"- **{label}**: {count}\n"
                    detection_stats.markdown(stats_text)
                
                # Small delay to prevent UI blocking
                time.sleep(0.01)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    finally:
        # Always release camera
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        st.session_state.is_running = False

else:
    # Show placeholder when not running
    FRAME_WINDOW.info("🎥 Click 'Detect Cameras' in sidebar, then click 'Start' to begin")
    fps_text.markdown("**⚡ FPS:** --")
    detection_text.markdown("**🎯 Detections:** 0")
    camera_status.markdown("**📹** --")
    detection_stats.markdown("### Current Frame\nNo detections yet")

# Footer
st.markdown("---")
st.markdown(
    "Built with Streamlit • FastAPI • ONNX Runtime | "
    "💡 Click 'Detect Cameras' to automatically find available cameras"
)