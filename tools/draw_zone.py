# tools/draw_zone.py
import cv2
import yaml
import numpy as np
import os

points = []
current_image = None
zone_saved = False

def mouse_callback(event, x, y, flags, param):
    global points, current_image, zone_saved
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(current_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(current_image, str(len(points)), (x+5, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Draw Polygon", current_image)
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        # Close and save polygon
        cv2.polylines(current_image, [np.array(points)], True, (0, 0, 255), 2)
        cv2.imshow("Draw Polygon", current_image)
        zone_saved = False
        while not zone_saved:
            zone_name = input("Enter zone name: ").strip()
            if not zone_name:
                print("Zone name cannot be empty.")
                continue
            required_ppe = input("Required PPE (comma separated, e.g., helmet,vest): ").strip()
            if not required_ppe:
                required_ppe = []
            else:
                required_ppe = [p.strip() for p in required_ppe.split(',')]
            trigger_vlm = input("Trigger VLM? (y/n): ").strip().lower() == 'y'
            
            new_zone = {
                "name": zone_name,
                "vertices": [list(p) for p in points],
                "required_ppe": required_ppe,
                "trigger_vlm": trigger_vlm
            }
            # Load existing config or create new
            config_path = "config/risk_zones.yaml"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    data = yaml.safe_load(f) or {"zones": []}
            else:
                data = {"zones": []}
            data["zones"].append(new_zone)
            with open(config_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"Zone '{zone_name}' saved to {config_path}")
            zone_saved = True
        points = []
        cv2.destroyWindow("Draw Polygon")

def main():
    global current_image
    # ---------- CHANGE THIS TO YOUR VIDEO FILE ----------
    video_path = r"D:\miners.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    # Read one frame to draw zones on
    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame")
        return
    current_image = frame.copy()
    cv2.imshow("Draw Polygon", current_image)
    cv2.setMouseCallback("Draw Polygon", mouse_callback)
    print("Instructions:")
    print("  - Left click: add point (vertices in order)")
    print("  - Right click: close polygon and save zone")
    print("  - Press 'q' to quit without saving")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()