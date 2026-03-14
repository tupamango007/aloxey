# main.py (расширенная версия)
import os
import cv2
from hough_gui import hough_gui_analysis
from hough_utils import quick_analysis, load_params_from_file
from barometer_reader import calibrate_gauge, load_calibration, find_needle_angle, angle_to_value, draw_needle, draw_reference_axis

def main():
    print("="*60)
    print("DETECTION OF LINES AND CIRCLES WITH HOUGH TRANSFORM")
    print("="*60)
    while True:
        print("\nMAIN MENU")
        print("1. Parameter tuning (GUI)")
        print("2. Quick analysis (default params)")
        print("3. Quick analysis with saved params")
        print("4. Barometer calibration (click center, min, max)")
        print("5. Read barometer from video (camera)")
        print("6. Read barometer from image file")
        print("7. Exit")
        choice = input("Choose option (1-7): ").strip()

        if choice == "1":
            path = input("Image path: ").strip()
            if os.path.exists(path):
                hough_gui_analysis(path)
            else:
                print("File not found.")
        elif choice == "2":
            path = input("Image path: ").strip()
            if os.path.exists(path):
                quick_analysis(path)
            else:
                print("File not found.")
        elif choice == "3":
            path = input("Image path: ").strip()
            if not os.path.exists(path):
                print("File not found.")
                continue
            params_file = input("Parameters file path: ").strip()
            if not os.path.exists(params_file):
                print("Parameters file not found.")
                continue
            try:
                params = load_params_from_file(params_file)
                print(f"Loaded parameters: {params}")
                quick_analysis(path, params)
            except Exception as e:
                print(f"Error loading parameters: {e}")
        elif choice == "4":
            path = input("Image path for calibration: ").strip()
            if os.path.exists(path):
                calib_file = input("Output calibration file (default: gauge_calib.json): ").strip()
                if not calib_file:
                    calib_file = 'gauge_calib.json'
                calibrate_gauge(path, calib_file)
            else:
                print("File not found.")

        elif choice == "5":
            calib_file = input("Calibration file (e.g., gauge_calib.json): ").strip()
            if not os.path.exists(calib_file):
                print("Calibration file not found.")
                continue
            calib_data = load_calibration(calib_file)
            
            hough_params = None
            params_file = input("Hough parameters file (optional, press Enter to skip): ").strip()
            if params_file and os.path.exists(params_file):
                hough_params = load_params_from_file(params_file)
                print(f"Loaded Hough parameters: {hough_params}")
            
            source = input("Use camera (0) or video file path? (Enter 0 for camera, or path to video): ").strip()
            if source == "0":
                cap = cv2.VideoCapture(0)
            else:
                if not os.path.exists(source):
                    print("Video file not found.")
                    continue
                cap = cv2.VideoCapture(source)
            
            if not cap.isOpened():
                print("Cannot open source")
                continue
            
            print("Press 'q' to quit, 's' to save current frame")
            print("Dynamic center enabled: program will re-detect circle on each frame.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or cannot read frame.")
                    break
                
                angle, tip, current_center = find_needle_angle(frame, calib_data, hough_params, dynamic_center=True)
                
                
                draw_reference_axis(frame, current_center, calib_data['radius'])
                
                if angle is not None:
                    value = angle_to_value(angle, calib_data)
                    if value is not None:
                        cv2.line(frame, current_center, tip, (0,0,255), 3)
                        cv2.circle(frame, current_center, 5, (255,0,0), -1)
                        cv2.putText(frame, f"{value} MPa", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        cv2.putText(frame, f"{angle:.1f}°", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                    else:
                        cv2.putText(frame, "Out of range", (10,30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                else:
                    cv2.putText(frame, "No needle", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                
                cv2.imshow('Barometer reader', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite('barometer_snapshot.jpg', frame)
                    print("Snapshot saved")
            
            cap.release()
            cv2.destroyAllWindows()

        elif choice == "6":
            
            path = input("Image path: ").strip()
            if not os.path.exists(path):
                print("File not found.")
                continue
            calib_file = input("Calibration file: ").strip()
            if not os.path.exists(calib_file):
                print("Calibration file not found.")
                continue
            calib_data = load_calibration(calib_file)
            
            hough_params = None
            params_file = input("Hough parameters file (optional, press Enter to skip): ").strip()
            if params_file and os.path.exists(params_file):
                hough_params = load_params_from_file(params_file)
                print(f"Loaded Hough parameters: {hough_params}")
            
            frame = cv2.imread(path)
           
            angle, tip, current_center = find_needle_angle(frame, calib_data, hough_params, dynamic_center=True)
            
            draw_reference_axis(frame, current_center, calib_data['radius'])
            
            if angle is not None:
                print(f"Detected angle: {angle:.2f}°")
                value = angle_to_value(angle, calib_data)
                if value is not None:
                    print(f"Value: {value} MPa")
                    cv2.line(frame, current_center, tip, (0,0,255), 3)
                    cv2.circle(frame, current_center, 5, (255,0,0), -1)
                    cv2.putText(frame, f"{value} MPa", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(frame, f"{angle:.1f}°", (10,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
                    cv2.imshow('Result', frame)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                else:
                    print("Angle is outside scale range")
            else:
                print("Needle not found")
        elif choice == "7":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
