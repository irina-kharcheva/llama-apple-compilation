# Qwen-1.5B Model Compilation for Apple Devices
# Компиляция модели Qwen-1.5B для устройств Apple

## English

This project provides tools for compiling the Qwen-1.5B model for Apple devices using CoreML.

### Prerequisites

- Python 3.8+
- macOS with Xcode installed
- Apple device for testing (iPhone/iPad/Mac)

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Compile the model:
```bash
python compile_qwen.py
```

2. Test and compare models:
```bash
python test_models.py
```

### Project Structure

- `compile_qwen.py` - Script for model compilation
- `test_models.py` - Script for testing and comparing models
- `requirements.txt` - Project dependencies
- `Qwen1_5B.mlpackage` - Compiled model (generated after running compile_qwen.py)

### Notes

- The compilation process includes 4-bit quantization for model optimization
- The compiled model is saved as a CoreML package
- Testing script compares both original and compiled model performance

## Russian

Этот проект предоставляет инструменты для компиляции модели Qwen-1.5B для устройств Apple с использованием CoreML.

### Предварительные требования

- Python 3.8+
- macOS с установленным Xcode
- Устройство Apple для тестирования (iPhone/iPad/Mac)

### Установка

1. Клонируйте этот репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

### Использование

1. Компиляция модели:
```bash
python compile_qwen.py
```

2. Тестирование и сравнение моделей:
```bash
python test_models.py
```

### Структура проекта

- `compile_qwen.py` - Скрипт для компиляции модели
- `test_models.py` - Скрипт для тестирования и сравнения моделей
- `requirements.txt` - Зависимости проекта
- `Qwen1_5B.mlpackage` - Скомпилированная модель (создается после запуска compile_qwen.py)

### Примечания

- Процесс компиляции включает 4-битную квантизацию для оптимизации модели
- Скомпилированная модель сохраняется в формате CoreML
- Скрипт тестирования сравнивает производительность оригинальной и скомпилированной моделей 
>>>>>>> 3a3ed40 (Initial commit)
