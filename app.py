# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# from datetime import datetime
# import pandas as pd
# import queue
# import threading

# # Streamlit UI setup
# st.set_page_config(page_title="CCTV Attendance System", layout="wide")
# st.title("📹 CCTV-Based Face Recognition Attendance")

# # Paths
# KNOWN_FACES_DIR = 'known_faces'
# ATTENDANCE_CSV = 'attendance.csv'
# os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# # Load known face encodings and names
# @st.cache_resource
# def load_known_faces():
#     images = []
#     classNames = []
#     for filename in os.listdir(KNOWN_FACES_DIR):
#         img_path = os.path.join(KNOWN_FACES_DIR, filename)
#         img = cv2.imread(img_path)
#         if img is not None:
#             images.append(img)
#             classNames.append(os.path.splitext(filename)[0])

#     known_encodings = []
#     for img in images:
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         enc = face_recognition.face_encodings(rgb)
#         if enc:
#             known_encodings.append(enc[0])
#     return known_encodings, classNames

# known_encodings, classNames = load_known_faces()

# # Attendance function
# def mark_attendance(name):
#     now = datetime.now()
#     date = now.strftime('%Y-%m-%d')
#     time = now.strftime('%H:%M:%S')

#     if not os.path.exists(ATTENDANCE_CSV):
#         df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
#         df.to_csv(ATTENDANCE_CSV, index=False)

#     df = pd.read_csv(ATTENDANCE_CSV)
#     if not ((df['Name'] == name) & (df['Date'] == date)).any():
#         df = pd.concat([df, pd.DataFrame([[name, date, time]], columns=['Name', 'Date', 'Time'])])
#         df.to_csv(ATTENDANCE_CSV, index=False)
#         st.success(f"✅ Attendance marked for {name} at {time}")

# # Video processing in a separate thread
# frame_queue = queue.Queue(maxsize=10)
# stop_event = threading.Event()

# def video_processing_thread(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         st.error("❌ Failed to open video stream. Check the RTSP URL or camera credentials.")
#         return
    
#     try:
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("⚠️ Failed to fetch frame from stream.")
#                 break
            
#             # Skip frames if queue is full to prevent lag
#             if frame_queue.full():
#                 continue
                
#             # Process frame
#             small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#             faces = face_recognition.face_locations(rgb_small)
#             encodes = face_recognition.face_encodings(rgb_small, faces)

#             for encode_face, face_loc in zip(encodes, faces):
#                 matches = face_recognition.compare_faces(known_encodings, encode_face)
#                 face_dist = face_recognition.face_distance(known_encodings, encode_face)
#                 match_index = np.argmin(face_dist)

#                 if matches[match_index]:
#                     name = classNames[match_index].upper()
#                     y1, x2, y2, x1 = [v * 4 for v in face_loc]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     mark_attendance(name)
            
#             # Put processed frame in queue
#             frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     finally:
#         cap.release()

# # Streamlit UI
# # rtsp_url = "rtsp://admin:going2KOKAN!@192.168.0.102:554/ch0_0.264"
# rtsp_url = "rtmp://live.restream.io/live/re_9645823_6708955baaf204d73ebc"
# start = st.button("▶️ Start Attendance")
# stop = st.button("⏹️ Stop Attendance")
# frame_display = st.empty()

# if start:
#     stop_event.clear()
#     processing_thread = threading.Thread(target=video_processing_thread, args=(rtsp_url,))
#     processing_thread.start()
    
#     st.info("Press Stop or Refresh to end the session.")
    
#     try:
#         while not stop_event.is_set():
#             if not frame_queue.empty():
#                 frame_display.image(frame_queue.get(), channels="RGB")
#     except:
#         stop_event.set()
#         processing_thread.join()

# if stop:
#     stop_event.set()




# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# from datetime import datetime
# import pandas as pd
# import queue
# import threading
# from PIL import Image

# # Custom CSS for styling
# def local_css(file_name):
#     try:
#         with open(file_name) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#     except:
#         st.markdown("""
#         <style>
#         /* Fallback basic styling */
#         .stApp {
#             background-color: #f5f7fa;
#         }
#         h1 {
#             color: #2c3e50;
#         }
#         .video-container {
#             border: 2px solid #3498db;
#             border-radius: 10px;
#             padding: 10px;
#         }
#         </style>
#         """, unsafe_allow_html=True)

# # Streamlit UI setup
# st.set_page_config(
#     page_title="CCTV Attendance System",
#     layout="wide",
#     page_icon="👥"
# )

# # Load custom CSS
# local_css("style.css")

# # App header
# col1, col2 = st.columns([1, 4])

# with col2:
#     st.title("Smart Attendance System")
#     st.markdown("""
#     <div class="subheader">
#     Real-time face recognition for automated attendance tracking
#     </div>
#     """, unsafe_allow_html=True)

# # Create a thread-safe way to update Streamlit elements
# class StreamlitState:
#     def __init__(self):
#         self.status = None
#         self.notifications = []
#         self.lock = threading.Lock()
        
#     def update_status(self, message, type="info"):
#         with self.lock:
#             self.status = (message, type)
            
#     def add_notification(self, message, type="info"):
#         with self.lock:
#             self.notifications.append((message, type))
            
#     def get_updates(self):
#         with self.lock:
#             status = self.status
#             notifications = self.notifications.copy()
#             self.notifications.clear()
#             return status, notifications

# state = StreamlitState()

# # Sidebar for controls
# with st.sidebar:
#     st.header("⚙️ System Controls")
    
#     rtsp_url = st.text_input(
#         "Camera Stream URL",
#         value="rtmp://live.restream.io/live/re_9645823_6708955baaf204d73ebc",
#         # value="rtsp://admin:going2KOKAN!@192.168.0.102:554/ch0_0.264",
#         help="Enter the RTSP/RTMP URL of your CCTV camera"
#     )
    
#     col1, col2 = st.columns(2)
#     with col1:
#         start = st.button("▶️ Start Monitoring", key="start")
#     with col2:
#         stop = st.button("⏹️ Stop Monitoring", key="stop")
    
#     st.markdown("---")
    
#     st.subheader("📊 System Status")
#     status_placeholder = st.empty()
    
#     st.subheader("📝 Today's Attendance")
#     if os.path.exists('attendance.csv'):
#         today = datetime.now().strftime('%Y-%m-%d')
#         df = pd.read_csv('attendance.csv')
#         today_df = df[df['Date'] == today]
#         st.dataframe(today_df, height=200)
#     else:
#         st.info("No attendance records yet for today")

# # Main content
# tab1, tab2 = st.tabs(["🎥 Live Feed", "📊 Reports"])

# with tab1:
#     st.markdown("""
#     <div class="video-container">
#     """, unsafe_allow_html=True)
    
#     frame_display = st.empty()
    
#     st.markdown("""
#     </div>
#     """, unsafe_allow_html=True)
    
#     st.subheader("🔔 Notifications")
#     notifications_placeholder = st.empty()

# with tab2:
#     if os.path.exists('attendance.csv'):
#         df = pd.read_csv('attendance.csv')
        
#         st.subheader("📅 Attendance Summary")
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Records", len(df))
#         with col2:
#             unique_people = df['Name'].nunique()
#             st.metric("Unique People", unique_people)
#         with col3:
#             today_count = len(df[df['Date'] == datetime.now().strftime('%Y-%m-%d')])
#             st.metric("Today's Attendance", today_count)
        
#         st.subheader("📋 Detailed Records")
#         date_range = st.date_input(
#             "Select date range",
#             value=[datetime.now().date(), datetime.now().date()],
#             max_value=datetime.now().date()
#         )
        
#         if len(date_range) == 2:
#             filtered_df = df[
#                 (df['Date'] >= str(date_range[0])) & 
#                 (df['Date'] <= str(date_range[1]))
#             ]
#             st.dataframe(filtered_df, use_container_width=True)
#     else:
#         st.info("No attendance records available yet")

# # Face recognition setup
# KNOWN_FACES_DIR = 'known_faces'
# ATTENDANCE_CSV = 'attendance.csv'
# os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# @st.cache_resource
# def load_known_faces():
#     images = []
#     classNames = []
#     for filename in os.listdir(KNOWN_FACES_DIR):
#         img_path = os.path.join(KNOWN_FACES_DIR, filename)
#         img = cv2.imread(img_path)
#         if img is not None:
#             images.append(img)
#             classNames.append(os.path.splitext(filename)[0])

#     known_encodings = []
#     for img in images:
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         enc = face_recognition.face_encodings(rgb)
#         if enc:
#             known_encodings.append(enc[0])
#     return known_encodings, classNames

# known_encodings, classNames = load_known_faces()

# # Thread-safe attendance marking
# def mark_attendance(name):
#     now = datetime.now()
#     date = now.strftime('%Y-%m-%d')
#     time = now.strftime('%H:%M:%S')

#     if not os.path.exists(ATTENDANCE_CSV):
#         df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
#         df.to_csv(ATTENDANCE_CSV, index=False)

#     df = pd.read_csv(ATTENDANCE_CSV)
#     if not ((df['Name'] == name) & (df['Date'] == date)).any():
#         df = pd.concat([df, pd.DataFrame([[name, date, time]], columns=['Name', 'Date', 'Time'])])
#         df.to_csv(ATTENDANCE_CSV, index=False)
#         state.add_notification(f"✅ {name} marked present at {time}", "success")

# # Video processing thread
# frame_queue = queue.Queue(maxsize=10)
# stop_event = threading.Event()

# def video_processing_thread(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         state.update_status("❌ Failed to connect to camera stream", "error")
#         return
    
#     state.update_status("🟢 System active - Monitoring in progress", "success")
    
#     try:
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 state.update_status("⚠️ Stream interruption - trying to reconnect", "warning")
#                 continue
            
#             if frame_queue.full():
#                 continue
                
#             small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#             faces = face_recognition.face_locations(rgb_small)
#             encodes = face_recognition.face_encodings(rgb_small, faces)

#             for encode_face, face_loc in zip(encodes, faces):
#                 matches = face_recognition.compare_faces(known_encodings, encode_face)
#                 face_dist = face_recognition.face_distance(known_encodings, encode_face)
#                 match_index = np.argmin(face_dist)

#                 if matches[match_index]:
#                     name = classNames[match_index].upper()
#                     y1, x2, y2, x1 = [v * 4 for v in face_loc]
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                     cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#                     mark_attendance(name)
            
#             frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     finally:
#         cap.release()
#         if not stop_event.is_set():
#             state.update_status("🔴 System stopped", "info")

# # Main execution loop
# if start:
#     stop_event.clear()
#     processing_thread = threading.Thread(target=video_processing_thread, args=(rtsp_url,))
#     processing_thread.daemon = True  # Make thread a daemon so it exits when main thread exits
#     processing_thread.start()

# if stop:
#     stop_event.set()
#     state.update_status("🟠 Stopping system...", "info")

# # Update UI elements in the main thread
# while True:
#     status_update, new_notifications = state.get_updates()
    
#     if status_update:
#         message, type = status_update
#         if type == "success":
#             status_placeholder.success(message)
#         elif type == "error":
#             status_placeholder.error(message)
#         elif type == "warning":
#             status_placeholder.warning(message)
#         else:
#             status_placeholder.info(message)
    
#     if new_notifications:
#         notifications_text = "\n\n".join([msg for msg, _ in new_notifications])
#         notifications_placeholder.info(notifications_text)
    
#     if not frame_queue.empty():
#         frame_display.image(frame_queue.get(), channels="RGB")
    
#     # Small delay to prevent high CPU usage
#     threading.Event().wait(0.1)



import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import pandas as pd
import threading
import queue
import time

# Directories and constants
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_CSV = 'attendance.csv'
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Streamlit config
st.set_page_config(page_title="Face Attendance", layout="wide")

# Load known faces
@st.cache_resource
def load_known_faces():
    images, classNames, known_encodings = [], [], []
    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            images.append(img)
            classNames.append(os.path.splitext(filename)[0])
            known_encodings.append(encodings[0])
    return known_encodings, classNames

known_encodings, classNames = load_known_faces()

# Attendance marker
def mark_attendance(name):
    now = datetime.now()
    date, time_now = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')
    if not os.path.exists(ATTENDANCE_CSV):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
    else:
        df = pd.read_csv(ATTENDANCE_CSV)

    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame([[name, date, time_now]], columns=["Name", "Date", "Time"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_CSV, index=False)

# Streamlit UI
st.title("📷 Face Recognition Attendance")
rtsp_url = st.text_input("Camera RTSP/RTMP URL", value="rtmp://live.restream.io/live/re_9645823_6708955baaf204d73ebc")

start_button = st.button("Start")
stop_button = st.button("Stop")

frame_placeholder = st.empty()
status_placeholder = st.empty()

frame_queue = queue.Queue()
stop_event = threading.Event()

def run_recognition(rtsp_url):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        status_placeholder.error("Failed to connect to camera.")
        return

    status_placeholder.success("Connected to camera. Monitoring...")
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_small)
        encodings = face_recognition.face_encodings(rgb_small, faces)

        for encode, face_loc in zip(encodings, faces):
            matches = face_recognition.compare_faces(known_encodings, encode)
            face_distances = face_recognition.face_distance(known_encodings, encode)
            if matches:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = classNames[best_match]
                    mark_attendance(name)
                    y1, x2, y2, x1 = [v * 4 for v in face_loc]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass
        frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    status_placeholder.info("Stopped monitoring.")

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if start_button:
    if not st.session_state.monitoring:
        stop_event.clear()
        threading.Thread(target=run_recognition, args=(rtsp_url,), daemon=True).start()
        st.session_state.monitoring = True

if stop_button:
    stop_event.set()
    st.session_state.monitoring = False

if st.session_state.monitoring and not frame_queue.empty():
    frame = frame_queue.get()
    frame_placeholder.image(frame, channels="RGB")

# Attendance display
st.subheader("📝 Attendance Records")
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
    st.dataframe(df)
else:
    st.info("No attendance recorded yet.")
