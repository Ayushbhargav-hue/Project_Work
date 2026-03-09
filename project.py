import cv2
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import os
import time
from datetime import datetime
import json

# --- Setup ---

# Download YOLOv8 face detection model
model_path = r"\model.pt"
face_detector = YOLO(model_path)

# Safe zone fraction (center rectangle)
SAFE_ZONE_FRACTION = 0.5  # 50% center area

# Log file for test sessions
LOG_FILE = "test_sessions.json"

# --- Helper functions ---

def load_or_create_log():
    """Load existing test sessions or create new log"""
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {"sessions": []}

def save_test_session(session_data):
    """Save test session data to log"""
    log = load_or_create_log()
    log["sessions"].append(session_data)
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def draw_safe_zone(frame, zone_coords):
    """Draw the safe zone rectangle"""
    x1, y1, x2, y2 = zone_coords
    # Draw semi-transparent overlay for safe zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    # Draw border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(frame, "SAFE ZONE", (x1 + 10, y1 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def calculate_face_position(face_box, zone_coords):
    """Check if face is completely inside safe zone"""
    fx1, fy1, fx2, fy2 = face_box
    zx1, zy1, zx2, zy2 = zone_coords
    
    # Check if face is fully inside zone
    if fx1 >= zx1 and fx2 <= zx2 and fy1 >= zy1 and fy2 <= zy2:
        return "inside"
    
    # Check if face is partially outside
    if fx2 < zx1 or fx1 > zx2 or fy2 < zy1 or fy1 > zy2:
        return "completely_outside"
    else:
        return "partially_outside"

def create_test_folder():
    """Create folder for this test session"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"test_session_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, timestamp

# --- Main monitoring program ---

print("=== Unknown Person Test Monitoring System ===")
print("This system will monitor a test candidate")
print("Rules:")
print("- Candidate's face must be visible at all times")
print("- Face must stay within the green safe zone")
print("- Test terminates immediately if face is missing or leaves zone")
print("- Multiple faces (invigilators) are allowed - test continues")
print("- Press 'q' to manually end the test")
print("-" * 50)

# Create session folder
session_folder, session_id = create_test_folder()
print(f"Session ID: {session_id}")
print(f"Saving evidence to: {session_folder}/")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Test state
test_active = True
termination_reason = None
violation_detected = False
frame_count = 0
candidate_face_tracked = False
last_candidate_box = None

# Session data
session_data = {
    "session_id": session_id,
    "start_time": datetime.now().isoformat(),
    "end_time": None,
    "duration_seconds": 0,
    "termination_reason": None,
    "violations": []
}

print("\nTest started! Monitoring candidate...")
print("Note: Multiple faces allowed (invigilators won't terminate test)")

while test_active:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
    
    frame_count += 1
    h, w, _ = frame.shape
    
    # Define safe zone (centered)
    zone_w = int(w * SAFE_ZONE_FRACTION)
    zone_h = int(h * SAFE_ZONE_FRACTION)
    zone_x1 = (w - zone_w) // 2
    zone_y1 = (h - zone_h) // 2
    zone_x2 = zone_x1 + zone_w
    zone_y2 = zone_y1 + zone_h
    zone_coords = (zone_x1, zone_y1, zone_x2, zone_y2)
    
    # Draw safe zone
    draw_safe_zone(frame, zone_coords)
    
    # Detect faces
    results = face_detector(frame)
    boxes = results[0].boxes
    
    # Status display
    status_color = (0, 255, 0)  # Green by default
    status_text = "Test in progress"
    
    # Find the candidate face (closest to center of safe zone)
    candidate_face_found = False
    candidate_box = None
    
    if boxes is not None and len(boxes) > 0:
        # Get all face boxes
        face_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Calculate face center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Calculate distance to safe zone center
            safe_zone_center_x = (zone_x1 + zone_x2) // 2
            safe_zone_center_y = (zone_y1 + zone_y2) // 2
            distance = ((center_x - safe_zone_center_x) ** 2 + (center_y - safe_zone_center_y) ** 2) ** 0.5
            
            face_boxes.append({
                "box": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "distance": distance
            })
        
        # Sort by distance to safe zone center (closest is likely the candidate)
        face_boxes.sort(key=lambda x: x["distance"])
        
        # The closest face to center is considered the candidate
        candidate_box_info = face_boxes[0]
        candidate_box = candidate_box_info["box"]
        candidate_face_found = True
        
        # Draw all faces with different colors
        for i, face_info in enumerate(face_boxes):
            x1, y1, x2, y2 = face_info["box"]
            
            if i == 0:  # Candidate face (closest to center)
                # Check candidate face position relative to safe zone
                position_status = calculate_face_position(candidate_box, zone_coords)
                
                if position_status == "inside":
                    box_color = (0, 255, 0)  # Green
                    label = "CANDIDATE"
                elif position_status == "partially_outside":
                    # Immediate termination for partial exit
                    cv2.putText(frame, "VIOLATION: Candidate partially outside zone!", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "TEST TERMINATED - PARTIAL EXIT", (w//2 - 250, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    termination_reason = "partial_exit"
                    violation_detected = True
                    
                    # Save evidence
                    filename = f"{session_folder}/partial_exit_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    # Log violation
                    violation = {
                        "time": datetime.now().isoformat(),
                        "type": "partial_exit",
                        "frame": frame_count
                    }
                    session_data["violations"].append(violation)
                    
                    print(f"\nVIOLATION: Candidate partially left safe zone! Test terminated.")
                    print(f"Evidence saved: {filename}")
                    break
                    
                else:  # completely_outside
                    # Immediate termination for complete exit
                    cv2.putText(frame, "VIOLATION: Candidate completely outside zone!", (20, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, "TEST TERMINATED - OUT OF ZONE", (w//2 - 250, h//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    termination_reason = "out_of_zone"
                    violation_detected = True
                    
                    # Save evidence
                    filename = f"{session_folder}/out_of_zone_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    print(f"\nVIOLATION: Candidate left safe zone! Test terminated.")
                    print(f"Evidence saved: {filename}")
                    break
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # Add face position indicator
                face_center = ((x1 + x2)//2, (y1 + y2)//2)
                cv2.circle(frame, face_center, 5, box_color, -1)
                
            else:  # Other faces (invigilators, etc.) - just mark them
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan, thin line
                cv2.putText(frame, "OTHER", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # If we broke out due to violation, end test
        if violation_detected:
            break
    
    # Check if no candidate face found
    if not candidate_face_found:
        # No face detected (candidate missing) - TERMINATE IMMEDIATELY
        cv2.putText(frame, "VIOLATION: Candidate not visible!", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "TEST TERMINATED - CANDIDATE MISSING", (w//2 - 300, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        termination_reason = "candidate_missing"
        violation_detected = True
        
        # Save evidence
        filename = f"{session_folder}/candidate_missing_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        
        # Log violation
        violation = {
            "time": datetime.now().isoformat(),
            "type": "candidate_missing",
            "frame": frame_count
        }
        session_data["violations"].append(violation)
        
        print(f"\nVIOLATION: Candidate not visible! Test terminated immediately.")
        print(f"Evidence saved: {filename}")
        break
    
    # Add status and timer
    elapsed_time = time.time() - session_start_time if 'session_start_time' in locals() else 0
    cv2.putText(frame, f"Status: {status_text}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    cv2.putText(frame, f"Time: {int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}", 
                (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show face counts
    total_faces = 0 if boxes is None else len(boxes)
    cv2.putText(frame, f"Total faces: {total_faces}", (w - 150, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    if total_faces > 1:
        cv2.putText(frame, f"(Invigilators present)", (w - 150, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to end test", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Show frame
    cv2.imshow("Test Monitoring System", frame)
    
    # Initialize session start time on first valid frame
    if frame_count == 1:
        session_start_time = time.time()
    
    # Exit on 'q' (manual termination)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        termination_reason = "manual_termination"
        print("\nTest manually terminated by operator.")
        break

# --- Test ended - save results ---
cap.release()
cv2.destroyAllWindows()

# Calculate session duration
if 'session_start_time' in locals():
    end_time = time.time()
    duration = end_time - session_start_time
    
    # Update session data
    session_data["end_time"] = datetime.now().isoformat()
    session_data["duration_seconds"] = round(duration, 2)
    session_data["termination_reason"] = termination_reason
    
    # Save session data
    save_test_session(session_data)
    
    # Create summary report
    report_path = f"{session_folder}/summary.txt"
    with open(report_path, 'w') as f:
        f.write("=== TEST SESSION SUMMARY ===\n")
        f.write(f"Session ID: {session_id}\n")
        f.write(f"Start Time: {session_data['start_time']}\n")
        f.write(f"End Time: {session_data['end_time']}\n")
        f.write(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
        f.write(f"Termination Reason: {termination_reason}\n")
        f.write(f"Total Violations: {len(session_data['violations'])}\n")
        
        if termination_reason in ["candidate_missing", "out_of_zone", "partial_exit"]:
            f.write(f"\nRESULT: FAILED - {termination_reason.replace('_', ' ').title()}\n")
        elif termination_reason == "manual_termination":
            f.write("\nRESULT: COMPLETED - Test ended normally by operator\n")
    
    print(f"\n=== TEST SUMMARY ===")
    print(f"Session ID: {session_id}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"Termination: {termination_reason}")
    print(f"Violations: {len(session_data['violations'])}")
    print(f"Report saved to: {report_path}")
    print(f"Evidence saved to: {session_folder}/")
