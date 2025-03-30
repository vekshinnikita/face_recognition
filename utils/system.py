import os
from pathlib import Path 

def ensure_directory_exists_os(path_to_file):
    """Проверяет, существует ли директория для сохранения файла, и создает ее при необходимости."""
    directory = os.path.dirname(path_to_file)  # Получаем путь к директории из пути к файлу
    if not os.path.exists(directory):
        os.makedirs(directory)  # Создает директорию и все необходимые родительские директории
        print(f"Создана директория: {directory}")
    else:
        print(f"Директория уже существует: {directory}")
        
        
def delete_file_os(file_path):
  """Удаляет файл по указанному пути, используя модуль os."""
  try:
    os.remove(file_path)  # Основная функция для удаления файла
    print(f"Файл '{file_path}' успешно удален.")
  except FileNotFoundError:
    print(f"Файл '{file_path}' не найден.")
  except PermissionError:
    print(f"Нет прав для удаления файла '{file_path}'.")
  except Exception as e:
    print(f"Произошла ошибка при удалении файла '{file_path}': {e}")


def delete_files_with_prefix(folder_path, prefix):
    """Удаляет все файлы в указанной папке, имена которых начинаются с заданного префикса, используя модуль pathlib."""
    try:
        folder = Path(folder_path)

        if not folder.is_dir():
            print(f"Папка '{folder_path}' не существует.")
            return

        files_to_delete = list(folder.glob(f"{prefix}*")) # Создаем список объектов Path, соответствующих шаблону

        if not files_to_delete:
            print(f"В папке '{folder_path}' не найдены файлы, начинающиеся с '{prefix}'.")
            return

        for file_path_obj in files_to_delete:
            try:
                file_path_obj.unlink()  # Удаляем файл
                print(f"Файл '{file_path_obj}' успешно удален.")
            except FileNotFoundError:
                print(f"Файл '{file_path_obj}' не найден (возможно, был удален ранее).")
            except PermissionError:
                print(f"Нет прав для удаления файла '{file_path_obj}'.")
            except Exception as e:
                print(f"Произошла ошибка при удалении файла '{file_path_obj}': {e}")

    except Exception as e:
        print(f"Произошла общая ошибка: {e}")