from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, TimestampType
from PIL import Image
import numpy as np
import random
import io
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from datetime import datetime

# --- ТОЧНО ТАКАЯ ЖЕ КОНФИГУРАЦИЯ КАК В РАБОЧЕМ СТРИМЕ ---
spark = SparkSession.builder.appName("MinIO-ETL-Batch").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- MinIO config (точно как в рабочем коде) ---
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# --- Пути ---
RAW_PATH = "s3a://raw/images_parquet/"
STAGED_PATH = "s3a://staged/images_parquet/"
CHECKPOINT_TABLE_PATH = "s3a://staged/processed_files_tracking/"

# --- Такая же схема как в рабочем коде ---
schema = StructType([StructField("image", BinaryType(), True)])

tracking_schema = StructType([
    StructField("file_path", StringType(), False),
    StructField("processed_at", TimestampType(), False),
    StructField("batch_id", StringType(), True),
    StructField("records_processed", StringType(), True)
])


def get_processed_files():
    """Читает таблицу с уже обработанными файлами"""
    try:
        # Пробуем прочитать tracking таблицу
        tracking_df = spark.read.schema(tracking_schema).parquet(CHECKPOINT_TABLE_PATH)
        processed_files = set(row.file_path for row in tracking_df.select("file_path").collect())
        print(f"📋 Найдено {len(processed_files)} ранее обработанных файлов")
        return processed_files
    except Exception:
        print("📋 Таблица отслеживания не найдена, начинаем с чистого листа")
        return set()


def save_processed_files(file_paths, batch_id, total_records):
    """Сохраняет информацию об обработанных файлах"""
    if not file_paths:
        return
    
    tracking_data = [(
        fp, 
        datetime.now(), 
        batch_id, 
        str(total_records)
    ) for fp in file_paths]
    
    tracking_df = spark.createDataFrame(tracking_data, schema=tracking_schema)
    tracking_df.coalesce(1).write.mode("append").parquet(CHECKPOINT_TABLE_PATH)
    print(f"💾 Сохранена информация о {len(file_paths)} обработанных файлах")


def get_unprocessed_files():
    """Возвращает список непрочитанных parquet файлов"""
    try:
        # Получаем список ВСЕХ файлов через Spark SQL (самый надежный способ)
        all_files_df = spark.sql(f"SELECT input_file_name() as file_path FROM parquet.`{RAW_PATH}`")
        all_files = [row.file_path for row in all_files_df.distinct().collect()]
        
        print(f"📁 Всего найдено {len(all_files)} parquet файлов в raw")
        
        # Получаем уже обработанные файлы
        processed_files = get_processed_files()
        
        # Фильтруем только новые файлы
        unprocessed_files = [f for f in all_files if f not in processed_files]
        
        print(f"🆕 Новых файлов для обработки: {len(unprocessed_files)}")
        
        if unprocessed_files:
            print("📄 Примеры новых файлов:")
            for f in unprocessed_files[:3]:
                print(f"  - {f}")
        
        return unprocessed_files
        
    except Exception as e:
        print(f"❌ Ошибка при получении списка файлов: {e}")
        return []


# --- ТА ЖЕ САМАЯ функция обработки изображений ---
def process_images(image_list, crop_mode='center', use_enhance=True,
                   apply_equalize=False, apply_white_balance=False,
                   apply_color_norm=False):
    """
    Приведение к 256x256, улучшение качества и минимальная аугментация.
    """
    target_size = (256, 256)
    processed = []

    # Статистики ImageNet для нормализации
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

    for img in image_list:
        # 1. Приведение к RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 2. Случайное кадрирование
        if crop_mode == 'center':
            ratio = max(target_size[0] / img.width, target_size[1] / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            left = (img_resized.width - target_size[0]) // 2
            top = (img_resized.height - target_size[1]) // 2
            img_cropped = img_resized.crop((left, top, left + target_size[0], top + target_size[1]))
        elif crop_mode == 'random':
            scale = random.uniform(0.8, 1.0)
            new_size = (int(target_size[0] / scale), int(target_size[1] / scale))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            left = random.randint(0, img_resized.width - target_size[0])
            top = random.randint(0, img_resized.height - target_size[1])
            img_cropped = img_resized.crop((left, top, left + target_size[0], top + target_size[1]))
        else:
            raise ValueError("crop_mode должен быть 'center' или 'random'")

        # 3. Блок улучшений
        img_enhanced = img_cropped.copy()

        if use_enhance:
            img_enhanced = ImageOps.autocontrast(img_enhanced, cutoff=2)
            img_enhanced = img_enhanced.filter(ImageFilter.MedianFilter(size=3))
            enhancer = ImageEnhance.Color(img_enhanced)
            img_enhanced = enhancer.enhance(1.1)

        if apply_equalize:
            img_enhanced = ImageOps.equalize(img_enhanced)

        if apply_white_balance:
            img_np = np.array(img_enhanced).astype(np.float32)
            mean_r = img_np[:,:,0].mean()
            mean_g = img_np[:,:,1].mean()
            mean_b = img_np[:,:,2].mean()
            avg = (mean_r + mean_g + mean_b) / 3.0
            img_np[:,:,0] = np.clip(img_np[:,:,0] * (avg / mean_r), 0, 255)
            img_np[:,:,1] = np.clip(img_np[:,:,1] * (avg / mean_g), 0, 255)
            img_np[:,:,2] = np.clip(img_np[:,:,2] * (avg / mean_b), 0, 255)
            img_enhanced = Image.fromarray(img_np.astype(np.uint8))

        if apply_color_norm:
            img_np = np.array(img_enhanced).astype(np.float32) / 255.0
            img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
            img_np = np.clip((img_np * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).astype(np.uint8)
            img_enhanced = Image.fromarray(img_np)

        # 4. Минимальная аугментация
        img_aug = img_enhanced.copy()
        if random.random() > 0.5:
            img_aug = ImageOps.mirror(img_aug)
        angle = random.uniform(-10, 10)
        if angle != 0:
            img_aug = img_aug.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(0,0,0))
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(img_aug)
            img_aug = enhancer.enhance(random.uniform(0.8, 1.2))

        processed.append(img_aug)

    print(f"✅ Обработано {len(processed)} изображений (размер 256x256).")
    return processed


def process_batch_files(file_paths, batch_id):
    """Обрабатывает группу файлов (почти как process_batch в стриме)"""
    
    if not file_paths:
        print(f"⚠️ Batch {batch_id}: нет файлов для обработки")
        return 0
    
    print(f"\n{'='*50}")
    print(f"📦 Batch {batch_id}: обработка {len(file_paths)} файлов")
    
    try:
        # Читаем файлы - ТОЧНО ТАК ЖЕ как в стриме, но batch
        df = spark.read.schema(schema).parquet(*file_paths)
        
        # Дальше ТОЧНО ТАК ЖЕ как в process_batch из стрима
        rows = df.collect()
        
        if not rows:
            print(f"Batch {batch_id} пуст, пропускаем")
            save_processed_files(file_paths, batch_id, 0)
            return 0

        pil_images = []
        
        for idx, row in enumerate(rows):
            try:
                img_bytes = row.image
                img = Image.open(io.BytesIO(img_bytes))
                pil_images.append(img)
            except Exception as e:
                print(f"Ошибка при загрузке изображения {idx}: {e}")
                continue

        if not pil_images:
            print(f"Нет валидных изображений в батче {batch_id}")
            save_processed_files(file_paths, batch_id, 0)
            return 0

        processed_images = process_images(
            pil_images,
            crop_mode='random',
            use_enhance=True,
            apply_equalize=False,
            apply_white_balance=False,
            apply_color_norm=False
        )

        processed_rows = []
        for img in processed_images:
            img_bytes_io = io.BytesIO()
            img.save(img_bytes_io, format='JPEG', quality=95)
            processed_rows.append((img_bytes_io.getvalue(),))

        processed_df = spark.createDataFrame(processed_rows, schema=schema)
        processed_df.write.mode("append").parquet(STAGED_PATH)
        
        save_processed_files(file_paths, batch_id, len(processed_rows))
        
        print(f"✅ Batch {batch_id} обработан и сохранён: {len(processed_rows)} изображений")
        return len(processed_rows)
        
    except Exception as e:
        print(f"❌ Ошибка в batch {batch_id}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Основная функция для запуска из Airflow"""
    
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🚀 Запуск пакетной обработки {batch_id}")
    
    # Получаем список непрочитанных файлов
    unprocessed_files = get_unprocessed_files()
    
    if not unprocessed_files:
        print("✨ Нет новых файлов для обработки")
        return
    
    # Обрабатываем все файлы за один раз (как один большой батч)
    # Или можно разбить на части если файлов очень много
    BATCH_SIZE = 10  # Количество файлов на один батч
    
    total_processed = 0
    total_files = len(unprocessed_files)
    
    for i in range(0, total_files, BATCH_SIZE):
        batch_files = unprocessed_files[i:i + BATCH_SIZE]
        sub_batch_id = f"{batch_id}_part_{i//BATCH_SIZE}"
        
        try:
            processed_count = process_batch_files(batch_files, sub_batch_id)
            total_processed += processed_count
        except Exception as e:
            print(f"❌ Критическая ошибка в батче {sub_batch_id}: {e}")
            # Продолжаем обработку следующих батчей
    
    print(f"\n{'='*50}")
    print(f"🎉 Обработка завершена!")
    print(f"📊 Всего обработано файлов: {total_files}")
    print(f"📊 Всего обработано изображений: {total_processed}")


if __name__ == "__main__":
    main()