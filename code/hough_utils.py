import cv2
import numpy as np
import math
import os
from barometer_reader import compute_angle
from hough_config import DEFAULT_HOUGH_PARAMS

def load_params_from_file(params_file):
    params = {}
    with open(params_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                key, val = line.split('=', 1)
                params[key] = int(val)
    return params

def quick_analysis(image_path, params=None):
    """Quick analysis with optional custom parameters."""
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    if params is None:
        p = DEFAULT_HOUGH_PARAMS.copy()
    else:
        p = DEFAULT_HOUGH_PARAMS.copy()
        p.update(params)

    if params is not None:
        for k in DEFAULT_HOUGH_PARAMS:
            if k in params:
                DEFAULT_HOUGH_PARAMS[k] = params[k]
    p = DEFAULT_HOUGH_PARAMS

    edges = cv2.Canny(blur, p['canny1'], p['canny2'])
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, p['hough_thresh'],
                            minLineLength=p['min_line_len'],
                            maxLineGap=p['max_line_gap'])

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50,
                               param1=p['circle_param1'],
                               param2=p['circle_param2'],
                               minRadius=p['min_radius'],
                               maxRadius=p['max_radius'])

    result = img.copy()
    circles_int = None
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for c in circles_int[0, :]:
            cv2.circle(result, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(result, (c[0], c[1]), 2, (255, 0, 0), 3)

    lines_in_circle = 0
    angle_info = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            in_circle = False
            if circles_int is not None:
                for c in circles_int[0, :]:
                    cx, cy, r = c
                    d1 = math.hypot(x1-cx, y1-cy)
                    d2 = math.hypot(x2-cx, y2-cy)
                    if d1 <= r*1.1 or d2 <= r*1.1:
                        in_circle = True
                        lines_in_circle += 1
                        # Определяем остриё
                        tip = (x1, y1) if d1 > d2 else (x2, y2)
                        angle = compute_angle((cx, cy), tip)
                        angle_info.append({'center': (cx,cy), 'radius': r, 'angle': round(angle,1)})
                        break

            if in_circle and p.get('filter_by_circle', 1):
                color = (0, 0, 255)
                thickness = 3
            else:
                color = (255, 0, 0)
                thickness = 1
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)

    cv2.imshow('Result (quick analysis)', result)
    cv2.imshow('Edges', edges)
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("ANGLES OF LINES IN CIRCLES (0° = down, clockwise)")
    print("="*50)
    if angle_info:
        groups = {}
        for a in angle_info:
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
        print("No lines inside circles.")
