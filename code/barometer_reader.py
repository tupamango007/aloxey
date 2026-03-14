import cv2
import numpy as np
import math
import json
import os

def compute_angle(center, point):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = math.degrees(math.atan2(dx, -dy))
    if angle < 0:
        angle += 360
    return angle

def draw_reference_axis(frame, center, radius, color=(0, 255, 255), thickness=1):
    down_point = (center[0], center[1] + radius)
    cv2.line(frame, center, down_point, color, thickness)
    cv2.putText(frame, "0°", (down_point[0]-20, down_point[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def calibrate_gauge(image_path, output_calib_file='gauge_calib.json'):
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return None
    
    clone = img.copy()
    cv2.namedWindow('Calibration')
    cv2.imshow('Calibration', clone)
    
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(clone, str(len(points)), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
            cv2.imshow('Calibration', clone)
    
    cv2.setMouseCallback('Calibration', mouse_callback)
    print("Инструкция: кликните ЛКМ в следующем порядке:")
    print("1. Центр циферблата")
    print("2. Точка на шкале, соответствующая МИНИМАЛЬНОМУ значению (например, 0)")
    print("3. Точка на шкале, соответствующая МАКСИМАЛЬНОМУ значению (например, 1.5)")
    print("После трёх кликов нажмите ESC для завершения калибровки и сохранения.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 and len(points) >= 3:
            break
        elif key == 27:
            print("Нужно выбрать минимум 3 точки")
    
    cv2.destroyAllWindows()
    
    if len(points) < 3:
        print("Калибровка отменена")
        return None
    
    center = points[0]
    p_min = points[1]
    p_max = points[2]
    
    angle_min = compute_angle(center, p_min)
    angle_max = compute_angle(center, p_max)
    
    radius = int(math.hypot(p_max[0]-center[0], p_max[1]-center[1]))
    
    print(f"Угол для минимума: {angle_min:.1f}°")
    print(f"Угол для максимума: {angle_max:.1f}°")
    print(f"Радиус циферблата: {radius} пикс.")

    if angle_min > angle_max:
        print("Внимание: угол минимума больше угла максимума.")
        print("Возможно, шкала идёт против часовой стрелки, либо вы кликнули в обратном порядке.")
        resp = input("Поменять min и max местами? (y/n): ").strip().lower()
        if resp == 'y':
            angle_min, angle_max = angle_max, angle_min
            print("Порядок изменён.")
    
    min_val = float(input("Введите минимальное значение на шкале (например, 0): "))
    max_val = float(input("Введите максимальное значение на шкале (например, 1.5): "))
    
    calib_data = {
        'center': center,
        'radius': radius,
        'angle_min': angle_min,
        'angle_max': angle_max,
        'value_min': min_val,
        'value_max': max_val,
        'unit': 'MPa'  # можно запросить у пользователя
    }
    
    with open(output_calib_file, 'w') as f:
        json.dump(calib_data, f, indent=4)
    
    print(f"Калибровка сохранена в {output_calib_file}")
    return calib_data

def load_calibration(calib_file):
    with open(calib_file, 'r') as f:
        return json.load(f)

def find_gauge_center(frame, calib_data, hough_params=None):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    if hough_params is None:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=100, param2=20,
                                   minRadius=int(calib_data['radius']*0.8),
                                   maxRadius=int(calib_data['radius']*1.2))
    else:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                                   param1=hough_params.get('circle_param1', 100),
                                   param2=hough_params.get('circle_param2', 20),
                                   minRadius=int(calib_data['radius']*0.8),
                                   maxRadius=int(calib_data['radius']*1.2))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        best = min(circles[0, :], key=lambda c: math.hypot(c[0]-calib_data['center'][0], c[1]-calib_data['center'][1]))
        return (best[0], best[1])
    return None

def find_needle_angle(frame, calib_data, hough_params=None, dynamic_center=True):

    if dynamic_center:
        center = find_gauge_center(frame, calib_data, hough_params)
        if center is None:
            center = tuple(calib_data['center'])  # fallback
    else:
        center = tuple(calib_data['center'])
    
    radius = calib_data.get('radius', 100)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    if hough_params is None:
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    else:
        edges = cv2.Canny(blur, hough_params['canny1'], hough_params['canny2'])
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=hough_params['hough_thresh'],
                                minLineLength=hough_params['min_line_len'],
                                maxLineGap=hough_params['max_line_gap'])
    
    if lines is None:
        return None, None, center
    
    close_thresh = radius * 0.15
    far_thresh = radius * 0.4
    
    best_line = None
    best_score = float('inf')
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = math.hypot(x1 - center[0], y1 - center[1])
        d2 = math.hypot(x2 - center[0], y2 - center[1])
        close_dist = min(d1, d2)
        far_dist = max(d1, d2)
        
        if close_dist < close_thresh and far_dist > far_thresh:
            score = close_dist - far_dist
            if score < best_score:
                best_score = score
                best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        close_thresh = radius * 0.25
        far_thresh = radius * 0.3
        for line in lines:
            x1, y1, x2, y2 = line[0]
            d1 = math.hypot(x1 - center[0], y1 - center[1])
            d2 = math.hypot(x2 - center[0], y2 - center[1])
            close_dist = min(d1, d2)
            far_dist = max(d1, d2)
            if close_dist < close_thresh and far_dist > far_thresh:
                score = close_dist - far_dist
                if score < best_score:
                    best_score = score
                    best_line = (x1, y1, x2, y2)
    
    if best_line is None:
        return None, None, center
    
    x1, y1, x2, y2 = best_line
    d1 = math.hypot(x1 - center[0], y1 - center[1])
    d2 = math.hypot(x2 - center[0], y2 - center[1])
    tip = (x1, y1) if d1 > d2 else (x2, y2)
    
    angle = compute_angle(center, tip)
    return angle, tip, center

def angle_to_value(angle, calib_data):
    a_min = calib_data['angle_min']
    a_max = calib_data['angle_max']
    v_min = calib_data['value_min']
    v_max = calib_data['value_max']
    
    if a_max < a_min:  # шкала переходит через 0°
        if angle < a_min:
            angle += 360
        if angle < a_min or angle > a_max + 360:
            return None
        value = v_min + (angle - a_min) * (v_max - v_min) / (a_max + 360 - a_min)
    else:
        if angle < a_min or angle > a_max:
            return None
        value = v_min + (angle - a_min) * (v_max - v_min) / (a_max - a_min)
    
    return round(value, 3)

def draw_needle(frame, center, tip, value):
    cv2.line(frame, center, tip, (0, 0, 255), 3)
    cv2.circle(frame, center, 5, (255, 0, 0), -1)
    cv2.putText(frame, f"{value} MPa", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return frame
