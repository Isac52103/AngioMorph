import yaml
import json
import sys
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: Path) -> dict:
    """
    Читает настройки путей и параметров из конфигурационного YAML файла.

    Args:
        config_path (Path): Абсолютный или относительный путь до конфигурационного файла.

    Returns:
        dict: Словарь с настройками. Ожидаемые ключи: 'model_weights', 'data_yaml_path', 'output_json'.
    """
    try:
        with config_path.open('r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    except FileNotFoundError:
        print(f"Файл не найден: {config_path.absolute()}")
    
    except yaml.YAMLError as ex:
        print(f"Ошибка синтаксиса в YAML файле: {ex}")
    
    except KeyError as ex:
        print(f"Ошибка структуры конфига: {ex}")
    
    except ValueError as ex:
        print(f"Ошибка содержимого: {ex}")
    
    except PermissionError as ex:
        print(f"Ошибка доступа: {ex}")
    
    except Exception as ex:
        print(f"Непредвиденная ошибка: {ex}")


def run_evaluation(model_path: Path, data_yaml_path: Path):
    """
    Загружает модель YOLO и запускает процесс тестирования на указанном датасете.

    Args:
        model_path (Path): Путь к файлу с весами модели
        data_yaml_path (Path): Путь к YAML файлу датасета

    Returns:
        ultralytics.engine.results.Results: Объект с результатами тестирования, содержащий сырые метрики и тензоры.
        None: Функция ничего не возвращает. Возникла ошибка.
    """
    print(f"Начало тестирования: {model_path.name}")

    if not model_path.exists():
        print (f"Файл модели {model_path.name} не найден в папке\nТестирование остановлено")
        return None

    try:
        model = YOLO(str(model_path))
        results = model.val(data=str(data_yaml_path))

        print("\nТестирование завершено")
        return results
    
    except RuntimeError as ex:
        print(f"Сбой процесса: {ex}")
        return None        
    
    except Exception as ex:
        print(f"Непредвиденная ошибка: {ex}")
        return None


def create_report(raw_results, model_path: Path, output_path: Path) -> None:
    """
    Извлекает сегментационные метрики из сырого объекта результатов YOLO, 
    формирует словарь и сохраняет его в JSON файл.

    Args:
        raw_results (ultralytics.engine.results.Results): Объект с метриками, полученный из функции run_evaluation.
        model_path (Path): Путь к модели.
        output_path (Path): Путь, по которому будет сохранен итоговый файл metrics.json.

    Returns:
        None: Функция ничего не возвращает, результат - сохраненный файл на диске.
    """
    print(f"Подготовка к сохранению отчета {output_path.name}...")

    report = {
        "model_name": model_path.name,
        "speed_ms": {
            "preprocess": float(raw_results.speed['preprocess']),
            "inference": float(raw_results.speed['inference']),
            "postprocess": float(raw_results.speed['postprocess'])
        },
        "metrics_mask": {
            "mAP_50_95": float(raw_results.seg.map),
            "mAP_50": float(raw_results.seg.map50),
            "mean_precision": float(raw_results.seg.p.mean()),
            "mean_recall": float(raw_results.seg.r.mean()),
            "mean_f1_score" : float(raw_results.seg.f1.mean())
        },
        "metrics_per_class": {}
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    found_classes = raw_results.seg.ap_class_index
    class_names = raw_results.names

    for i in range(len(found_classes)):
        class_id = str(found_classes[i])

        report["metrics_per_class"][class_id] = {
            "class_name": class_names.get(int(class_id), f"unknown_{class_id}"),
            "mAP_50_95": float(raw_results.seg.maps[i]),
            "f1_score": float(raw_results.seg.f1[i]),
            "recall": float(raw_results.seg.r[i]),
            "precision": float(raw_results.seg.p[i])
        }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
                             
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)

        print(f"Файл {output_path.name} сохранен по пути {output_path.absolute()}")
    
    except TypeError as ex:
        print(f"Ошибка структуры данных: {ex}")
    
    except PermissionError as ex:
        print(f"Ошибка доступа: {ex}")
    
    except IsADirectoryError as ex:
        print(f"Конфликт имен: {ex}")
    
    except OSError as ex:
        print(f"Системная ошибка ввода-вывода: {ex}")
    
    except Exception as ex:
        print(f"Непредвиденная ошибка при сохранении отчета: {ex}")


if __name__ == "__main__":
    CONFIG_FILE = Path("configs/test_config.yaml")
    config = load_config(CONFIG_FILE)

    model_file = Path(config['model_weights'])
    data_file = Path(config['data_yaml_path'])
    val_folder = Path("data/images/val")
    report_file = Path(config['output_json'])

    yolo_metrics = run_evaluation(model_file, data_file)

    if yolo_metrics is None:
        print("Завершение - оценка YOLO провалилась")
        sys.exit(1)

    create_report(yolo_metrics, model_file, report_file)