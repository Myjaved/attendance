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






# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime
# import threading
# import queue
# import time

# st.set_page_config(page_title="Face Attendance", layout="wide")

# KNOWN_FACES_DIR = 'known_faces'
# ATTENDANCE_CSV = 'attendance.csv'
# os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# # Load known faces
# @st.cache_resource
# def load_known_faces():
#     encodings = []
#     names = []
#     for file in os.listdir(KNOWN_FACES_DIR):
#         img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
#         if img is not None:
#             rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             faces = face_recognition.face_encodings(rgb)
#             if faces:
#                 encodings.append(faces[0])
#                 names.append(os.path.splitext(file)[0])
#     return encodings, names

# known_encodings, known_names = load_known_faces()

# # Attendance logic
# def mark_attendance(name):
#     now = datetime.now()
#     date = now.strftime('%Y-%m-%d')
#     time_str = now.strftime('%H:%M:%S')
    
#     if not os.path.exists(ATTENDANCE_CSV):
#         df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
#         df.to_csv(ATTENDANCE_CSV, index=False)

#     df = pd.read_csv(ATTENDANCE_CSV)
#     if not ((df['Name'] == name) & (df['Date'] == date)).any():
#         new_entry = pd.DataFrame([[name, date, time_str]], columns=["Name", "Date", "Time"])
#         df = pd.concat([df, new_entry])
#         df.to_csv(ATTENDANCE_CSV, index=False)

# # Frame queue and stop event
# frame_queue = queue.Queue()
# stop_event = threading.Event()

# # Camera worker (runs in background thread)
# def camera_worker(rtsp_url):
#     cap = cv2.VideoCapture(rtsp_url)
#     while not stop_event.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         # Resize & detect faces
#         small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#         faces = face_recognition.face_locations(rgb_small)
#         encodes = face_recognition.face_encodings(rgb_small, faces)

#         for encode, face_loc in zip(encodes, faces):
#             matches = face_recognition.compare_faces(known_encodings, encode)
#             dist = face_recognition.face_distance(known_encodings, encode)
#             best = np.argmin(dist)
#             if matches[best]:
#                 name = known_names[best].upper()
#                 y1, x2, y2, x1 = [v * 4 for v in face_loc]
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#                 cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
#                 mark_attendance(name)

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         if not frame_queue.full():
#             frame_queue.put(rgb_frame)
#         time.sleep(0.03)

#     cap.release()

# # UI Sidebar
# st.sidebar.header("Controls")
# rtsp_url = st.sidebar.text_input("Camera URL", "rtmp://live.restream.io/live/re_9645823_6708955baaf204d73ebc")
# col1, col2 = st.sidebar.columns(2)

# if col1.button("▶️ Start"):
#     stop_event.clear()
#     threading.Thread(target=camera_worker, args=(rtsp_url,), daemon=True).start()

# if col2.button("⏹️ Stop"):
#     stop_event.set()

# # Live Feed Display (in main thread only)
# frame_area = st.empty()
# while True:
#     if not frame_queue.empty():
#         frame = frame_queue.get()
#         frame_area.image(frame, channels="RGB")
#     time.sleep(0.05)






# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# from datetime import datetime
# import pandas as pd
# import threading

# # Custom CSS for styling
# def local_css(file_name):
#     try:
#         with open(file_name) as f:
#             st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
#     except:
#         st.markdown("""
#         <style>
#         .stApp {
#             background-color: #f5f7fa;
#         }
#         h1 {
#             color: #2c3e50;
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

# # Create a thread-safe state management class
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

# # Main tabs
# tab1, tab2 = st.tabs(["📊 Reports", "📒 Logs"])

# with tab1:
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

# with tab2:
#     notifications_placeholder = st.empty()

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

# # Attendance marking
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

# # Thread setup
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
#                 state.update_status("⚠️ Stream interruption - reconnecting...", "warning")
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
#                     mark_attendance(name)

#     finally:
#         cap.release()
#         if not stop_event.is_set():
#             state.update_status("🔴 System stopped", "info")

# # Start/stop monitoring
# if 'monitoring_thread' not in st.session_state:
#     st.session_state.monitoring_thread = None

# if start:
#     stop_event.clear()
#     st.session_state.monitoring_thread = threading.Thread(target=video_processing_thread, args=(rtsp_url,))
#     st.session_state.monitoring_thread.daemon = True
#     st.session_state.monitoring_thread.start()
#     state.update_status("🟢 Monitoring started...", "success")

# if stop:
#     stop_event.set()
#     state.update_status("🔴 Monitoring stopped.", "info")

# # Update system status and logs
# status_update, new_notifications = state.get_updates()

# if status_update:
#     msg, typ = status_update
#     getattr(status_placeholder, typ)(msg)

# if new_notifications:
#     notifications_text = "\n\n".join([msg for msg, _ in new_notifications])
#     notifications_placeholder.info(notifications_text)



# in working
# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime
# import threading

# # === Constants ===
# KNOWN_FACES_DIR = 'known_faces'
# ATTENDANCE_CSV = 'attendance.csv'
# os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# # === Helper: Load known faces ===
# def load_known_faces():
#     images, names = [], []
#     for file in os.listdir(KNOWN_FACES_DIR):
#         img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
#         if img is not None:
#             images.append(img)
#             names.append(os.path.splitext(file)[0])
#     encodings = []
#     for img in images:
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         enc = face_recognition.face_encodings(rgb)
#         if enc:
#             encodings.append(enc[0])
#     return encodings, names

# # === Helper: Mark attendance ===
# def mark_attendance(name):
#     now = datetime.now()
#     date, time = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')

#     if not os.path.exists(ATTENDANCE_CSV):
#         pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(ATTENDANCE_CSV, index=False)

#     df = pd.read_csv(ATTENDANCE_CSV)
#     if not ((df['Name'] == name) & (df['Date'] == date)).any():
#         new_entry = pd.DataFrame([[name, date, time]], columns=['Name', 'Date', 'Time'])
#         df = pd.concat([df, new_entry])
#         df.to_csv(ATTENDANCE_CSV, index=False)
#         print(f"[INFO] Marked present: {name} at {time}")

# # === Video Thread ===
# stop_event = threading.Event()

# def video_thread(rtsp_url, encodings, names):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print("❌ Could not connect to stream.")
#         return
#     print("✅ Video stream opened.")

#     try:
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 continue

#             small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#             face_locations = face_recognition.face_locations(rgb_small)
#             face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

#             for face_encoding in face_encodings:
#                 matches = face_recognition.compare_faces(encodings, face_encoding)
#                 face_distances = face_recognition.face_distance(encodings, face_encoding)
#                 if matches:
#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = names[best_match_index].upper()
#                         mark_attendance(name)
#     finally:
#         cap.release()
#         print("🛑 Stream released.")

# # === Streamlit UI ===
# st.set_page_config("Smart Attendance", layout="wide")
# st.title("🎯 Smart Attendance System (RTSP Stream)")

# # Sidebar
# with st.sidebar:
#     st.header("Controls")
#     rtsp_url = st.text_input("RTSP/RTMP URL", value="rtmp://live.restream.io/live/re_9645823_6708955baaf204d73ebc")
#     start_btn = st.button("▶️ Start Monitoring")
#     stop_btn = st.button("⏹️ Stop Monitoring")

# # Load encodings once
# if "encodings" not in st.session_state:
#     st.session_state.encodings, st.session_state.names = load_known_faces()

# # Start Monitoring
# if start_btn:
#     if not stop_event.is_set():
#         stop_event.clear()
#         thread = threading.Thread(
#             target=video_thread,
#             args=(rtsp_url, st.session_state.encodings, st.session_state.names)
#         )
#         thread.daemon = True
#         thread.start()
#         st.success("🟢 Monitoring started!")

# # Stop Monitoring
# if stop_btn:
#     stop_event.set()
#     st.warning("🛑 Monitoring stopped.")

# # Attendance Report
# st.markdown("## 📅 Today's Attendance")
# if os.path.exists(ATTENDANCE_CSV):
#     df = pd.read_csv(ATTENDANCE_CSV)
#     today = datetime.now().strftime('%Y-%m-%d')
#     df_today = df[df["Date"] == today]
#     st.dataframe(df_today, height=300)
# else:
#     st.info("No attendance yet for today.")



# import streamlit as st
# import cv2
# import face_recognition
# import numpy as np
# import os
# import pandas as pd
# from datetime import datetime
# import threading

# # === Constants ===
# KNOWN_FACES_DIR = 'known_faces'
# ATTENDANCE_CSV = 'attendance.csv'
# os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# # === Helper: Load known faces ===
# def load_known_faces():
#     images, names = [], []
#     for file in os.listdir(KNOWN_FACES_DIR):
#         img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
#         if img is not None:
#             images.append(img)
#             names.append(os.path.splitext(file)[0])
#     encodings = []
#     for img in images:
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         enc = face_recognition.face_encodings(rgb)
#         if enc:
#             encodings.append(enc[0])
#     # Free memory
#     del images
#     return encodings, names

# # === Helper: Mark attendance ===
# def mark_attendance(name):
#     now = datetime.now()
#     date, time = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')

#     if not os.path.exists(ATTENDANCE_CSV):
#         pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(ATTENDANCE_CSV, index=False)

#     df = pd.read_csv(ATTENDANCE_CSV)
#     if not ((df['Name'] == name) & (df['Date'] == date)).any():
#         new_entry = pd.DataFrame([[name, date, time]], columns=['Name', 'Date', 'Time'])
#         df = pd.concat([df, new_entry])
#         df.to_csv(ATTENDANCE_CSV, index=False)
#         print(f"[INFO] Marked present: {name} at {time}")

# # === Video Thread ===
# stop_event = threading.Event()

# # def video_thread(rtsp_url, encodings, names):
# #     cap = cv2.VideoCapture(rtsp_url)
# #     if not cap.isOpened():
# #         print("❌ Could not connect to stream.")
# #         return
# #     print("✅ Video stream opened.")

# #     try:
# #         while not stop_event.is_set():
# #             ret, frame = cap.read()
# #             if not ret:
# #                 continue

# #             small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
# #             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

# #             face_locations = face_recognition.face_locations(rgb_small)
# #             face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

# #             for face_encoding in face_encodings:
# #                 matches = face_recognition.compare_faces(encodings, face_encoding)
# #                 face_distances = face_recognition.face_distance(encodings, face_encoding)
# #                 if matches:
# #                     best_match_index = np.argmin(face_distances)
# #                     if matches[best_match_index]:
# #                         name = names[best_match_index].upper()
# #                         mark_attendance(name)
# #     finally:
# #         cap.release()
# #         print("🛑 Stream released.")


# def video_thread(rtsp_url, encodings, names):
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print("❌ Could not connect to stream.")
#         return
#     print("✅ Video stream opened.")
#     frame_count = 0
#     try:
#         while not stop_event.is_set():
#             ret, frame = cap.read()
#             if not ret:
#                 continue
#             frame_count += 1
#             if frame_count % 10 != 0:  # process only every 10th frame
#                 continue

#             small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

#             face_locations = face_recognition.face_locations(rgb_small)
#             face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

#             for face_encoding in face_encodings:
#                 matches = face_recognition.compare_faces(encodings, face_encoding)
#                 face_distances = face_recognition.face_distance(encodings, face_encoding)
#                 if matches:
#                     best_match_index = np.argmin(face_distances)
#                     if matches[best_match_index]:
#                         name = names[best_match_index].upper()
#                         mark_attendance(name)
#     finally:
#         cap.release()
#         print("🛑 Stream released.")


# # === Streamlit UI ===
# st.set_page_config("Smart Attendance", layout="wide")
# st.title("🎯 Smart Attendance System (RTSP Stream)")

# # Sidebar
# with st.sidebar:
#     st.header("Controls")
#     rtsp_url = st.text_input("RTSP/RTMP URL", value="")
#     start_btn = st.button("▶️ Start Monitoring")
#     stop_btn = st.button("⏹️ Stop Monitoring")

# # Start Monitoring
# if start_btn:
#     with st.spinner("🔄 Loading known faces..."):
#         encodings, names = load_known_faces()

#     if len(encodings) == 0:
#         st.error("No valid face encodings found. Please add face images to `known_faces` folder.")
#     else:
#         if not stop_event.is_set():
#             stop_event.clear()
#             thread = threading.Thread(
#                 target=video_thread,
#                 args=(rtsp_url, encodings, names)
#             )
#             thread.daemon = True
#             thread.start()
#             st.success("🟢 Monitoring started!")

# # Stop Monitoring
# if stop_btn:
#     stop_event.set()
#     st.warning("🛑 Monitoring stopped.")

# # Attendance Report
# st.markdown("## 📅 Today's Attendance")
# if os.path.exists(ATTENDANCE_CSV):
#     df = pd.read_csv(ATTENDANCE_CSV)
#     today = datetime.now().strftime('%Y-%m-%d')
#     df_today = df[df["Date"] == today]
#     st.dataframe(df_today, height=300)
# else:
#     st.info("No attendance yet for today.")




import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
import threading
from queue import Queue
import time

# === Constants ===
KNOWN_FACES_DIR = 'known_faces'
ATTENDANCE_CSV = 'attendance.csv'
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# === Shared State ===
stop_event = threading.Event()
result_queue = Queue()

# === Helper: Load known faces ===
def load_known_faces():
    images, names = [], []
    for file in os.listdir(KNOWN_FACES_DIR):
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file))
        if img is not None:
            images.append(img)
            names.append(os.path.splitext(file)[0])
    encodings = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb)
        if enc:
            encodings.append(enc[0])
    del images
    return encodings, names

# === Helper: Mark attendance ===
def mark_attendance(name):
    now = datetime.now()
    date, time_str = now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S')

    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=['Name', 'Date', 'Time']).to_csv(ATTENDANCE_CSV, index=False)

    df = pd.read_csv(ATTENDANCE_CSV)
    if not ((df['Name'] == name) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame([[name, date, time_str]], columns=['Name', 'Date', 'Time'])
        df = pd.concat([df, new_entry])
        df.to_csv(ATTENDANCE_CSV, index=False)
        print(f"[INFO] Marked present: {name} at {time_str}")

# === Video Thread ===
def video_thread(rtsp_url, encodings, names):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("❌ Could not connect to stream.")
        return
    print("✅ Video stream opened.")
    frame_count = 0
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frame_count += 1
            if frame_count % 10 != 0:
                continue

            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small)
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(encodings, face_encoding)
                face_distances = face_recognition.face_distance(encodings, face_encoding)
                if matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = names[best_match_index].upper()
                        mark_attendance(name)
                        print(f"[INFO] Detected and marked: {name}")
                        result_queue.put(name)
    finally:
        cap.release()
        print("🛑 Stream released.")

# === Streamlit UI ===
st.set_page_config("Smart Attendance", layout="wide")
st.title("🎯 Smart Attendance System (RTMP Stream)")

# Sidebar
with st.sidebar:
    st.header("Controls")
    rtsp_url = st.text_input("RTSP/RTMP URL", value="")
    start_btn = st.button("▶️ Start Monitoring")
    stop_btn = st.button("⏹️ Stop Monitoring")

# Detected Name Area
detected_name_area = st.empty()

# UI Loop to display detection alerts
def show_detection_alerts():
    while not stop_event.is_set():
        try:
            name = result_queue.get(timeout=1)
            detected_name_area.success(f"✅ Face Detected: {name}")
        except:
            pass

# Start Monitoring
if start_btn:
    with st.spinner("🔄 Loading known faces..."):
        encodings, names = load_known_faces()

    if len(encodings) == 0:
        st.error("No valid face encodings found. Please add face images to `known_faces` folder.")
    else:
        if not stop_event.is_set():
            stop_event.clear()
            thread = threading.Thread(target=video_thread, args=(rtsp_url, encodings, names))
            thread.daemon = True
            thread.start()

            ui_thread = threading.Thread(target=show_detection_alerts)
            ui_thread.daemon = True
            ui_thread.start()

            st.success("🟢 Monitoring started!")

# Stop Monitoring
if stop_btn:
    stop_event.set()
    st.warning("🛑 Monitoring stopped.")

# Attendance Report
st.markdown("## 📅 Today's Attendance")
if os.path.exists(ATTENDANCE_CSV):
    df = pd.read_csv(ATTENDANCE_CSV)
    today = datetime.now().strftime('%Y-%m-%d')
    df_today = df[df["Date"] == today]
    st.dataframe(df_today, height=300)
else:
    st.info("No attendance yet for today.")
