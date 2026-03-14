# hough_config.py
# Общие параметры по умолчанию для всех модулей

DEFAULT_HOUGH_PARAMS = {
    'canny1': 158,
    'canny2': 200,
    'hough_thresh': 90,
    'min_line_len': 90,
    'max_line_gap': 15,
    'circle_param1': 100,
    'circle_param2': 56,       # более чувствительное значение для поиска кругов
    'min_radius': 135,
    'max_radius': 290,
    'filter_by_circle': 1
}