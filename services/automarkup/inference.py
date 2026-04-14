import os
import io
import sys
import logging
from pathlib import Path
import time

import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from minio.error import S3Error

# ---------- Настройка логирования ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("auto-markup")

# ---------- Конфигурация из переменных окружения ----------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

SOURCE_BUCKET = os.getenv("SOURCE_BUCKET", "staged")      # бакет с исходными parquet
TARGET_BUCKET = os.getenv("TARGET_BUCKET", "automarkup")  # бакет для масок
MODEL_TYPE = os.getenv("MODEL_TYPE", "mobilenet")         # "resnet50" или "mobilenet"
NUM_CLASSES = int(os.getenv("NUM_CLASSES", "11"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))            # размер батча для GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Путь к файлам моделей внутри контейнера
MODEL_DIR = Path("/app/")
MODEL_PATHS = {
    "mobilenet": MODEL_DIR / "DeepLabV3_MobileNetV3_best.pth",
}

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---------- Инициализация клиента MinIO ----------
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Проверка SOURCE_BUCKET (обязательно должен существовать)
if not minio_client.bucket_exists(SOURCE_BUCKET):
    logger.error(f"Source bucket '{SOURCE_BUCKET}' не существует. Завершение.")
    sys.exit(1)

# Проверка TARGET_BUCKET: если нет — создать
if not minio_client.bucket_exists(TARGET_BUCKET):
    try:
        minio_client.make_bucket(TARGET_BUCKET)
        logger.info(f"Target bucket '{TARGET_BUCKET}' успешно создан")
    except S3Error as e:
        logger.error(f"Не удалось создать bucket '{TARGET_BUCKET}': {e}")
        sys.exit(1)
else:
    logger.info(f"Target bucket '{TARGET_BUCKET}' уже существует")

# ---------- Загрузка модели ----------
def load_model(model_type="mobilenet", num_classes=11):
    logger.info(f"Загрузка модели {model_type} на устройство {DEVICE}")
    if model_type == "resnet50":
        model = models.segmentation.deeplabv3_resnet50(
            weights=None, pretrained_backbone=False, aux_loss=False
        )
    else:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, pretrained_backbone=False, aux_loss=False
        )

    # Замена классификатора
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)

    weights_path = MODEL_PATHS[model_type]
    if not weights_path.exists():
        logger.error(f"Файл весов не найден: {weights_path}")
        sys.exit(1)

    state_dict = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    logger.info("Модель загружена")
    return model

model = load_model(MODEL_TYPE, NUM_CLASSES)

# ---------- Функции обработки ----------
def decode_image_from_bytes(image_bytes):
    """Преобразует байты изображения в PIL.Image."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Ошибка декодирования изображения: {e}")
        return None

def predict_batch(images):
    """Принимает список PIL.Image, возвращает numpy‑массив масок [N, H, W]."""
    batch_tensors = []
    valid_indices = []
    for i, img in enumerate(images):
        if img is not None:
            try:
                batch_tensors.append(transform(img))
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Ошибка трансформации изображения {i}: {e}")

    if not batch_tensors:
        return np.array([]), []

    batch = torch.stack(batch_tensors).to(DEVICE)

    with torch.no_grad():
        outputs = model(batch)["out"]
        masks = outputs.argmax(1).cpu().numpy()  # [B, H, W]

    return masks, valid_indices

def process_parquet_file(object_name):
    """
    Обрабатывает один Parquet‑файл из SOURCE_BUCKET.
    Читает все строки, для каждой получает изображение, делает предсказание,
    формирует новый Parquet с колонками: image (байты исходного изображения),
    mask (байты PNG‑представления маски), class_ids (список классов в маске?).
    Сохраняет в TARGET_BUCKET с тем же именем.
    """
    logger.info(f"Обработка файла: {object_name}")

    # Чтение parquet из MinIO в память
    try:
        response = minio_client.get_object(SOURCE_BUCKET, object_name)
        data = response.read()
        response.close()
        response.release_conn()
    except S3Error as e:
        logger.error(f"Ошибка чтения объекта {object_name}: {e}")
        return

    # Чтение таблицы Parquet
    try:
        table = pq.read_table(io.BytesIO(data))
    except Exception as e:
        logger.error(f"Ошибка парсинга Parquet {object_name}: {e}")
        return

    # Проверяем наличие колонки "image"
    if "image" not in table.column_names:
        logger.warning(f"Колонка 'image' отсутствует в {object_name}, пропускаем")
        return

    image_bytes_list = table["image"].to_pylist()  # список байтов
    total_images = len(image_bytes_list)
    logger.info(f"Найдено {total_images} изображений")

    # Подготавливаем массивы для результатов
    result_images = []   # байты оригинальных изображений (можно не сохранять, если не нужно)
    result_masks = []    # байты масок в PNG
    result_meta = []     # например, исходный индекс или имя файла

    # Обработка батчами
    for start_idx in range(0, total_images, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, total_images)
        batch_bytes = image_bytes_list[start_idx:end_idx]

        # Декодируем изображения
        pil_images = [decode_image_from_bytes(b) for b in batch_bytes]

        # Предсказание
        masks, valid_indices = predict_batch(pil_images)
        if len(masks) == 0:
            continue

        # Для каждого валидного изображения сохраняем маску как PNG в байтах
        for local_idx, global_idx_offset in enumerate(valid_indices):
            global_idx = start_idx + global_idx_offset
            original_bytes = batch_bytes[global_idx_offset]
            mask_array = masks[local_idx].astype(np.uint8)

            # Конвертируем маску в байты PNG
            mask_img = Image.fromarray(mask_array, mode="L")
            buf = io.BytesIO()
            mask_img.save(buf, format="PNG")
            mask_bytes = buf.getvalue()

            result_images.append(original_bytes)
            result_masks.append(mask_bytes)
            # Можно добавить дополнительные метаданные
            result_meta.append({"source_file": object_name, "row_index": global_idx})

        logger.info(f"Обработано {end_idx}/{total_images}")

    # Создаём новую таблицу Parquet
    new_table = pa.table({
        "image": pa.array(result_images, type=pa.binary()),
        "mask_png": pa.array(result_masks, type=pa.binary()),
        "metadata": pa.array([str(m) for m in result_meta], type=pa.string())
    })

    # Сохраняем в MinIO (в TARGET_BUCKET)
    out_buffer = io.BytesIO()
    pq.write_table(new_table, out_buffer, compression="snappy")
    out_buffer.seek(0)

    try:
        minio_client.put_object(
            TARGET_BUCKET,
            object_name,
            out_buffer,
            length=out_buffer.getbuffer().nbytes,
            content_type="application/parquet"
        )
        logger.info(f"Файл {object_name} успешно сохранён в {TARGET_BUCKET}")
    except S3Error as e:
        logger.error(f"Ошибка сохранения {object_name}: {e}")

def list_parquet_files(bucket, prefix=""):
    """Рекурсивно возвращает список имён объектов с расширением .parquet."""
    objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
    return [obj.object_name for obj in objects if obj.object_name.endswith(".parquet")]

def main():
    logger.info("Запуск авторазметки")
    parquet_files = list_parquet_files(SOURCE_BUCKET)
    logger.info(f"Найдено {len(parquet_files)} parquet‑файлов для обработки")

    for pf in parquet_files:
        try:
            process_parquet_file(pf)
        except Exception as e:
            logger.exception(f"Критическая ошибка при обработке {pf}: {e}")
            # Продолжаем со следующим файлом

    logger.info("Авторазметка завершена")

if __name__ == "__main__":
    main()