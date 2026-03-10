# diagnostic.py
import cv2
import numpy as np
import os

print("ДИАГНОСТИКА ИЗОБРАЖЕНИЯ")
print("=" * 60)

# Проверяем существование файла
image_path = "test.jpg"
if not os.path.exists(image_path):
    print(f"Файл {image_path} не найден!")
    exit()

# Загружаем изображение
img = cv2.imread(image_path)
if img is None:
    print(f"Не удалось загрузить {image_path} как изображение")
    
    # Попробуем прочитать как бинарный файл
    with open(image_path, 'rb') as f:
        data = f.read()
    print(f"Размер файла: {len(data)} байт")
    print(f"Первые 100 байт: {data[:100]}")
    exit()

print(f"Успешно загружено!")
print(f"Размер: {img.shape}")
print(f"Тип данных: {img.dtype}")
print(f"Диапазон значений: {img.min()} - {img.max()}")

# Показываем изображение
cv2.imshow('Original', img)

# Конвертируем в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Пробуем разные пороги
_, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Binary (normal)', thresh1)
cv2.imshow('Binary (inverted)', thresh2)

print("\nНажмите любую клавишу для продолжения...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Проверяем, есть ли контуры
contours, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nНайдено контуров: {len(contours)}")

# Рисуем контуры
result = img.copy()
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    print(f"Контур {i}: x={x}, y={y}, w={w}, h={h}, площадь={cv2.contourArea(cnt)}")
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Contours', result)
print("\nНажмите любую клавишу для выхода...")
cv2.waitKey(0)
cv2.destroyAllWindows()