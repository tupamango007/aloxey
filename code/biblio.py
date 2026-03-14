# install_dependencies.py
import subprocess
import sys

def install_packages():
    packages = [
        'opencv-python',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pillow'
    ]
    
    print("Установка необходимых библиотек...")
    
    for package in packages:
        print(f"Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Пробуем установить TensorFlow или PyTorch
    print("\nПопытка установить TensorFlow...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        print("TensorFlow успешно установлен!")
    except:
        print("Не удалось установить TensorFlow. Устанавливаю PyTorch...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
            print("PyTorch успешно установлен!")
        except:
            print("Не удалось установить TensorFlow или PyTorch.")
            print("Будет использована упрощенная версия с KNN.")
    
    print("\nВсе библиотеки установлены!")

if __name__ == "__main__":
    install_packages()