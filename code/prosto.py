import cv2
import numpy as np
import os
import math

def main_analysis(image_path):
    """Основная функция анализа"""
    
    # 1. Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение")
        return
    
    # 2. Обрабатываем
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Находим линии
    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 50, 10)
    
    # 4. Находим круги
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, 50, 
                              param1=200, param2=80, 
                              minRadius=100, maxRadius=200)
    
    # 5. Рисуем результат
    result = img.copy()
    
    # Рисуем линии
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Рисуем круги
    if circles is not None:
        circles_int = np.uint16(np.around(circles))
        for circle in circles_int[0, :]:
            cv2.circle(result, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            cv2.circle(result, (circle[0], circle[1]), 2, (255, 0, 0), 3)
    
    # 6. Анализируем линии внутри кругов
    if circles is not None and lines is not None:
        circles_int = np.uint16(np.around(circles))
        
        print("\n" + "=" * 50)
        print("АНАЛИЗ ЛИНИЙ В КРУГАХ:")
        print("=" * 50)
        
        for i, circle in enumerate(circles_int[0, :]):
            center_x, center_y, radius = circle
            
            print(f"\nКруг {i+1}:")
            print(f"  Центр: ({center_x}, {center_y})")
            print(f"  Радиус: {radius}")
            
            angles = []
            
            # Проверяем каждую линию
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Расстояние от концов линии до центра круга
                dist1 = math.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
                dist2 = math.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
                
                # Если линия внутри круга
                if dist1 <= radius or dist2 <= radius:
                    # Вычисляем угол
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    if angle < 0:
                        angle += 180
                    angles.append(angle)
            
            if angles:
                print(f"  Линий в круге: {len(angles)}")
                print(f"  Углы: {[round(a, 1) for a in angles]}")
            else:
                print("  Нет линий в круге")
    
    # 7. Показываем результат
    cv2.imshow('Исходное изображение', img)
    cv2.imshow('Результат', result)
    cv2.imwrite('результат.jpg', result)
    
    print("\nНажмите любую клавишу для продолжения...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Запуск программы
if __name__ == "__main__":
    print("=" * 50)
    print("АНАЛИЗ ЛИНИЙ И КРУГОВ")
    print("=" * 50)
    
    image_path = input("Введите путь к изображению: ").strip()
    
    if os.path.exists(image_path):
        main_analysis(image_path)
    else:
        print("Файл не найден!")
