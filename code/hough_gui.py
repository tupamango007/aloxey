import cv2
import numpy as np
import math
import time
import os
from barometer_reader import compute_angle
from hough_config import DEFAULT_HOUGH_PARAMS


    # ... создание трекбаров ...
    # При создании трекбаров начальные значения берутся из params
    

def hough_gui_analysis(image_path):
    """Graphical interface for tuning Hough parameters and angle analysis."""
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return None, None, None

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Windows
    cv2.namedWindow('Hough Settings', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough Settings', 600, 400)
    cv2.resizeWindow('Result', 800, 600)
    cv2.resizeWindow('Edges', 400, 300)

    params = DEFAULT_HOUGH_PARAMS.copy()
    

    # Trackbars
    cv2.createTrackbar('Canny1', 'Hough Settings', params['canny1'], 500, lambda x: None)
    cv2.createTrackbar('Canny2', 'Hough Settings', params['canny2'], 500, lambda x: None)
    cv2.createTrackbar('Line threshold', 'Hough Settings', params['hough_thresh'], 500, lambda x: None)
    cv2.createTrackbar('Min length', 'Hough Settings', params['min_line_len'], 500, lambda x: None)
    cv2.createTrackbar('Max gap', 'Hough Settings', params['max_line_gap'], 100, lambda x: None)
    cv2.createTrackbar('Param1 (circle)', 'Hough Settings', params['circle_param1'], 500, lambda x: None)
    cv2.createTrackbar('Circle sensitivity', 'Hough Settings', params['circle_param2'], 100, lambda x: None)
    cv2.createTrackbar('Min radius', 'Hough Settings', params['min_radius'], 300, lambda x: None)
    cv2.createTrackbar('Max radius', 'Hough Settings', params['max_radius'], 500, lambda x: None)
    cv2.createTrackbar('Filter (on/off)', 'Hough Settings', params['filter_by_circle'], 1, lambda x: None)

    print("\n" + "="*60)
    print("HOUGH PARAMETER TUNING MODE")
    print("="*60)
    print("Red lines – inside circle, blue – outside.")
    print("Angles are measured from downward vertical (0° = down), clockwise.")
    print("After exit, angles of lines inside circles will be displayed.")
    print("ESC/q – quit, s – save result, p – save parameters to file.\n")

    final_angles = []
    final_circles = None
    final_lines = None
    final_params = params.copy()

    while True:
        # Read trackbar values
        params['canny1'] = max(1, cv2.getTrackbarPos('Canny1', 'Hough Settings'))
        params['canny2'] = max(params['canny1']+1, cv2.getTrackbarPos('Canny2', 'Hough Settings'))
        params['hough_thresh'] = max(1, cv2.getTrackbarPos('Line threshold', 'Hough Settings'))
        params['min_line_len'] = max(1, cv2.getTrackbarPos('Min length', 'Hough Settings'))
        params['max_line_gap'] = max(1, cv2.getTrackbarPos('Max gap', 'Hough Settings'))
        params['circle_param1'] = max(1, cv2.getTrackbarPos('Param1 (circle)', 'Hough Settings'))
        params['circle_param2'] = max(1, cv2.getTrackbarPos('Circle sensitivity', 'Hough Settings'))
        params['min_radius'] = max(1, cv2.getTrackbarPos('Min radius', 'Hough Settings'))
        params['max_radius'] = max(params['min_radius']+1, cv2.getTrackbarPos('Max radius', 'Hough Settings'))
        params['filter_by_circle'] = cv2.getTrackbarPos('Filter (on/off)', 'Hough Settings')

        # Edge detection
        edges = cv2.Canny(blur, params['canny1'], params['canny2'])

        # Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, params['hough_thresh'],
                                 minLineLength=params['min_line_len'],
                                 maxLineGap=params['max_line_gap'])

        # Circles
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=params['circle_param1'],
                                   param2=params['circle_param2'],
                                   minRadius=params['min_radius'],
                                   maxRadius=params['max_radius'])

        result = original.copy()
        circles_int = None
        if circles is not None:
            circles_int = np.uint16(np.around(circles))
            for c in circles_int[0, :]:
                cv2.circle(result, (c[0], c[1]), c[2], (0, 255, 0), 2)
                cv2.circle(result, (c[0], c[1]), 2, (255, 0, 0), 3)

        # Line analysis
        lines_in_circle_count = 0
        current_angles = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_in_circle = False
                if circles_int is not None:
                    for c in circles_int[0, :]:
                        cx, cy, r = c
                        d1 = math.hypot(x1-cx, y1-cy)
                        d2 = math.hypot(x2-cx, y2-cy)
                        if d1 <= r*1.1 or d2 <= r*1.1:   # small tolerance
                            line_in_circle = True
                            lines_in_circle_count += 1

                            # Определяем остриё (дальний от центра конец)
                            tip = (x1, y1) if d1 > d2 else (x2, y2)
                            # Вычисляем угол в единой системе (0° = вниз, по часовой)
                            angle = compute_angle((cx, cy), tip)

                            current_angles.append({
                                'center': (cx, cy),
                                'radius': r,
                                'angle': round(angle, 1)
                            })
                            break

                # Color
                if line_in_circle and params['filter_by_circle']:
                    color = (0, 0, 255)      # red
                    thickness = 3
                else:
                    color = (255, 0, 0)      # blue
                    thickness = 1
                cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        # Save final state
        final_params = params.copy()
        final_circles = circles
        final_lines = lines
        final_angles = current_angles

        # Display
        cv2.imshow('Result', result)
        cv2.imshow('Edges', edges)

        # Info panel
        info = np.zeros((100, 600, 3), dtype=np.uint8)
        lines_cnt = len(lines) if lines is not None else 0
        circles_cnt = len(circles_int[0]) if circles_int is not None else 0
        cv2.putText(info, f'Lines: {lines_cnt}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(info, f'Circles: {circles_cnt}', (200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(info, f'Lines inside: {lines_in_circle_count}', (400,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        filt = 'ON' if params['filter_by_circle'] else 'OFF'
        col = (0,255,0) if params['filter_by_circle'] else (0,0,255)
        cv2.putText(info, f'Filter: {filt}', (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow('Info', info)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('s'):
            t = int(time.time())
            cv2.imwrite(f'result_{t}.jpg', result)
            cv2.imwrite(f'edges_{t}.jpg', edges)
            print(f"Saved result_{t}.jpg")
        elif key == ord('p'):
            base = os.path.splitext(os.path.basename(image_path))[0]
            params_file = f'{base}_params.txt'
            with open(params_file, 'w') as f:
                f.write("# Hough parameters saved from GUI\n")
                for k, v in params.items():
                    f.write(f"{k}={v}\n")
            print(f"Parameters saved to {params_file}")

    cv2.destroyAllWindows()

    # Print angles after closing
    print("\n" + "="*60)
    print("ANGLES OF LINES INSIDE CIRCLES (0° = down, clockwise)")
    print("="*60)

    if final_angles:
        groups = {}
        for a in final_angles:
            key = f"{a['center']}_{a['radius']}"
            if key not in groups:
                groups[key] = {'center': a['center'], 'radius': a['radius'], 'angles': []}
            if a['angle'] not in groups[key]['angles']:
                groups[key]['angles'].append(a['angle'])

        total = 0
        for i, (_, g) in enumerate(groups.items()):
            if g['angles']:
                print(f"\nCircle {i+1}: center {g['center']}, radius {g['radius']}")
                print(f"  Lines: {len(g['angles'])}, angles: {sorted(g['angles'])}")
                total += len(g['angles'])
        print(f"\nTotal lines in circles: {total}")
    else:
        print("No lines found inside circles.")
        if final_circles is None:
            print("  - Circles were not detected.")
        if final_lines is None:
            print("  - Lines were not detected.")
        if final_params['filter_by_circle'] == 0:
            print("  - Filter was turned off.")

    return final_lines, final_circles, final_params