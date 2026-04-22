from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, BinaryType, StringType, TimestampType
from pyspark.sql.functions import input_file_name
import os
import hashlib
from datetime import datetime
import io
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm

spark = SparkSession.builder.appName("MinIO-AutoMarkup-ToLocal").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- MinIO config ---
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

# --- Пути ---
AUTOMARKUP_PATH = "s3a://automarkup/images_parquet/"
LOCAL_PATH = "/data/automarkup_photos/"
TRACKING_FILE = "/data/automarkup_photos/.downloaded_tracking.parquet"
CHECKPOINT_FILE = "/data/automarkup_photos/.last_batch.txt"

# --- Схемы ---
tracking_schema = StructType([
    StructField("file_hash", StringType(), False),
    StructField("local_path", StringType(), False),
    StructField("source_file", StringType(), True),
    StructField("downloaded_at", TimestampType(), False),
    StructField("image_size", StringType(), True),
    StructField("batch_id", StringType(), True)
])


def get_last_batch_id():
    """Получает ID последнего обработанного батча"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return f.read().strip()
    return None


def save_batch_id(batch_id):
    """Сохраняет ID обработанного батча"""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(batch_id)


def get_downloaded_hashes():
    """Читает информацию о уже скачанных изображениях"""
    if os.path.exists(TRACKING_FILE):
        try:
            tracking_df = spark.read.schema(tracking_schema).parquet(f"file://{TRACKING_FILE}")
            return set(row.file_hash for row in tracking_df.select("file_hash").collect())
        except:
            return set()
    return set()


def calculate_hash(image_bytes):
    """Вычисляет MD5 хеш изображения"""
    return hashlib.md5(image_bytes).hexdigest()


def save_tracking_info(downloaded_info):
    """Сохраняет информацию о скачанных изображениях"""
    if not downloaded_info:
        return
    
    tracking_data = [(
        info['hash'],
        info['local_path'],
        info['source_file'],
        info['downloaded_at'],
        info['size'],
        info['batch_id']
    ) for info in downloaded_info]
    
    tracking_df = spark.createDataFrame(tracking_data, schema=tracking_schema)
    tracking_df.write.mode("append").parquet(f"file://{TRACKING_FILE}")


def overlay_mask_on_image(image_bytes, mask_bytes, alpha=0.4):
    """Накладывает цветную маску на изображение"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
        image_array = np.array(image)
        mask_array = np.array(mask)
        
        mask_norm = mask_array / max(mask_array.max(), 1)
        colored_mask = cm.jet(mask_norm)[:, :, :3]
        colored_mask = (colored_mask * 255).astype(np.uint8)
        
        overlay = (1 - alpha) * image_array + alpha * colored_mask
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        result_img = Image.fromarray(overlay)
        buf = io.BytesIO()
        result_img.save(buf, format="JPEG", quality=95)
        
        return buf.getvalue()
    except:
        return None


def download_new_images():
    """Скачивает новые изображения с наложенными масками"""
    
    os.makedirs(LOCAL_PATH, exist_ok=True)
    
    # Читаем данные из MinIO
    try:
        df = spark.read.format("parquet").load(AUTOMARKUP_PATH)
    except:
        print("❌ Не удалось прочитать данные из automarkup")
        return 0
    
    # Проверяем наличие нужных колонок
    if "image" not in df.columns or "mask_png" not in df.columns:
        print("❌ Отсутствуют обязательные колонки image/mask_png")
        return 0
    
    # Добавляем источник и собираем данные
    df_with_source = df.withColumn("source_file", input_file_name())
    rows = df_with_source.collect()
    
    if not rows:
        print("📁 Нет данных для обработки")
        return 0
    
    downloaded_hashes = get_downloaded_hashes()
    current_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    new_images = []
    downloaded_info = []
    duplicate_in_batch = set()
    
    print(f"📁 Найдено записей: {len(rows)}")
    
    for row in rows:
        image_bytes = row["image"]
        mask_bytes = row["mask_png"]
        
        if not image_bytes or not mask_bytes:
            continue
        
        image_hash = calculate_hash(image_bytes)
        
        if image_hash in downloaded_hashes or image_hash in duplicate_in_batch:
            continue
        
        duplicate_in_batch.add(image_hash)
        
        filename = f"marked_{image_hash[:12]}.jpg"
        local_path = os.path.join(LOCAL_PATH, filename)
        
        if os.path.exists(local_path):
            downloaded_hashes.add(image_hash)
            continue
        
        new_images.append({
            'image_bytes': image_bytes,
            'mask_bytes': mask_bytes,
            'path': local_path,
            'hash': image_hash,
            'source_file': row["source_file"]
        })
    
    print(f"🆕 Новых изображений: {len(new_images)}")
    
    if not new_images:
        return 0
    
    # Обрабатываем и сохраняем
    for img_info in new_images:
        overlay_bytes = overlay_mask_on_image(
            img_info['image_bytes'],
            img_info['mask_bytes']
        )
        
        if overlay_bytes:
            with open(img_info['path'], "wb") as f:
                f.write(overlay_bytes)
            
            downloaded_info.append({
                'hash': img_info['hash'],
                'local_path': img_info['path'],
                'source_file': img_info['source_file'],
                'downloaded_at': datetime.now(),
                'size': str(len(overlay_bytes)),
                'batch_id': current_batch_id
            })
    
    if downloaded_info:
        save_tracking_info(downloaded_info)
        save_batch_id(current_batch_id)
        print(f"✅ Сохранено: {len(downloaded_info)} изображений")
    
    return len(downloaded_info)


def main():
    print(f"🚀 Загрузка размеченных изображений из {AUTOMARKUP_PATH}")
    
    try:
        count = download_new_images()
        if count == 0:
            print("✨ Нет новых изображений")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()

# from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, BinaryType, StringType, TimestampType
# from pyspark.sql.functions import input_file_name, col
# import os
# import hashlib
# from datetime import datetime
# import io
# import numpy as np
# from PIL import Image
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.cm as cm

# spark = SparkSession.builder.appName("MinIO-AutoMarkup-ToLocal").getOrCreate()
# spark.sparkContext.setLogLevel("WARN")

# # --- MinIO config ---
# hadoop_conf = spark._jsc.hadoopConfiguration()
# hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
# hadoop_conf.set("fs.s3a.access.key", "minioadmin")
# hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
# hadoop_conf.set("fs.s3a.path.style.access", "true")
# hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
# hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")

# # --- Пути ---
# AUTOMARKUP_PATH = "s3a://automarkup/"
# LOCAL_PATH = "/data/automarkup_photos/"
# TRACKING_FILE = "/data/automarkup_photos/.downloaded_tracking.parquet"
# CHECKPOINT_FILE = "/data/automarkup_photos/.last_batch.txt"

# # --- Схема для automarkup данных ---
# # Важно: укажите точную схему ваших данных
# automarkup_schema = StructType([
#     StructField("image", BinaryType(), True),
#     StructField("mask_png", BinaryType(), True),
#     StructField("metadata", StringType(), True)
# ])

# tracking_schema = StructType([
#     StructField("file_hash", StringType(), False),
#     StructField("local_path", StringType(), False),
#     StructField("source_file", StringType(), True),
#     StructField("downloaded_at", TimestampType(), False),
#     StructField("image_size", StringType(), True),
#     StructField("batch_id", StringType(), True)
# ])


# def diagnose_minio_access():
#     """Диагностика доступа к MinIO через Spark"""
#     print("\n🔍 ДИАГНОСТИКА ДОСТУПА К MINIO ЧЕРЕЗ SPARK:")
    
#     try:
#         # Проверяем доступ к bucket через Spark
#         print(f"Проверка пути: {AUTOMARKUP_PATH}")
        
#         # Пробуем прочитать схему
#         try:
#             df = spark.read.format("parquet").load(AUTOMARKUP_PATH)
#             print(f"✅ Успешное подключение к {AUTOMARKUP_PATH}")
#             return df
#         except Exception as e:
#             print(f"❌ Ошибка чтения из {AUTOMARKUP_PATH}: {e}")
            
#             # Пробуем альтернативные пути
#             alt_paths = [
#                 "s3a://automarkup/images_parquet/",
#                 "s3a://automarkup/images_parquet",
#                 "s3a://automarkup/*.parquet",
#             ]
            
#             for alt_path in alt_paths:
#                 try:
#                     print(f"\nПробуем альтернативный путь: {alt_path}")
#                     df = spark.read.format("parquet").load(alt_path)
#                     print(f"✅ Успешное чтение из {alt_path}")
#                     return df
#                 except Exception as e2:
#                     print(f"❌ Ошибка: {e2}")
            
#             # Проверяем через список файлов
#             print("\n🔍 Проверка через listFiles:")
#             try:
#                 # Используем правильный способ для S3
#                 files_df = spark.read.format("binaryFile").load("s3a://automarkup/")
#                 print(f"✅ Найдены файлы в bucket:")
#                 files_df.select("path").show(10, truncate=False)
                
#                 # Фильтруем только parquet файлы
#                 parquet_files = files_df.filter(col("path").endswith(".parquet"))
#                 parquet_count = parquet_files.count()
#                 print(f"📁 Найдено {parquet_count} parquet файлов")
                
#                 if parquet_count > 0:
#                     # Пробуем прочитать первый parquet файл
#                     first_file = parquet_files.select("path").first()[0]
#                     print(f"📄 Чтение первого файла: {first_file}")
#                     df = spark.read.parquet(first_file)
#                     return df
                    
#             except Exception as e3:
#                 print(f"❌ Ошибка при listFiles: {e3}")
                
#     except Exception as e:
#         print(f"❌ Ошибка диагностики: {e}")
        
#     return None


# def get_last_batch_id():
#     """Получает ID последнего обработанного батча"""
#     if os.path.exists(CHECKPOINT_FILE):
#         with open(CHECKPOINT_FILE, 'r') as f:
#             return f.read().strip()
#     return None


# def save_batch_id(batch_id):
#     """Сохраняет ID обработанного батча"""
#     with open(CHECKPOINT_FILE, 'w') as f:
#         f.write(batch_id)


# def get_downloaded_hashes():
#     """Читает информацию о уже скачанных изображениях"""
#     if os.path.exists(TRACKING_FILE):
#         try:
#             tracking_df = spark.read.schema(tracking_schema).parquet(f"file://{TRACKING_FILE}")
#             downloaded_hashes = set(row.file_hash for row in tracking_df.select("file_hash").collect())
#             print(f"📋 Найдено {len(downloaded_hashes)} ранее скачанных изображений")
#             return downloaded_hashes
#         except Exception as e:
#             print(f"⚠️ Ошибка чтения tracking файла: {e}")
#             return set()
#     else:
#         print("📋 Tracking файл не найден, начинаем с чистого листа")
#         return set()


# def calculate_hash(image_bytes):
#     """Вычисляет MD5 хеш изображения"""
#     return hashlib.md5(image_bytes).hexdigest()


# def save_tracking_info(downloaded_info):
#     """Сохраняет информацию о скачанных изображениях"""
#     if not downloaded_info:
#         return
    
#     tracking_data = [(
#         info['hash'],
#         info['local_path'],
#         info['source_file'],
#         info['downloaded_at'],
#         info['size'],
#         info['batch_id']
#     ) for info in downloaded_info]
    
#     tracking_df = spark.createDataFrame(tracking_data, schema=tracking_schema)
#     tracking_df.write.mode("append").parquet(f"file://{TRACKING_FILE}")
#     print(f"💾 Обновлен tracking файл: добавлено {len(downloaded_info)} записей")


# def overlay_mask_on_image(image_bytes, mask_bytes, alpha=0.4):
#     """Накладывает цветную маску на изображение"""
#     try:
#         # Декодируем изображение и маску
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#         mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        
#         # Приводим размеры к одному
#         if mask.size != image.size:
#             mask = mask.resize(image.size, Image.Resampling.NEAREST)
        
#         # Конвертируем в numpy
#         image_array = np.array(image)
#         mask_array = np.array(mask)
        
#         # Нормализуем маску
#         if mask_array.max() > 0:
#             mask_norm = mask_array / mask_array.max()
#         else:
#             mask_norm = mask_array
        
#         # Применяем цветовую карту Jet
#         colored_mask = cm.jet(mask_norm)[:, :, :3]
#         colored_mask = (colored_mask * 255).astype(np.uint8)
        
#         # Накладываем с прозрачностью
#         overlay = (1 - alpha) * image_array + alpha * colored_mask
#         overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
#         # Сохраняем как JPEG
#         result_img = Image.fromarray(overlay)
#         buf = io.BytesIO()
#         result_img.save(buf, format="JPEG", quality=95)
        
#         return buf.getvalue()
        
#     except Exception as e:
#         print(f"❌ Ошибка наложения маски: {e}")
#         return None


# def process_dataframe(df):
#     """Обрабатывает DataFrame с данными из automarkup"""
    
#     # Проверяем схему
#     print("\n📊 Схема данных:")
#     df.printSchema()
    
#     # Проверяем наличие нужных колонок
#     required_columns = ["image", "mask_png"]
#     available_columns = df.columns
    
#     print(f"\n📋 Доступные колонки: {available_columns}")
    
#     missing_columns = [col for col in required_columns if col not in available_columns]
#     if missing_columns:
#         print(f"❌ Отсутствуют колонки: {missing_columns}")
#         return 0
    
#     # Добавляем колонку с источником
#     df_with_source = df.withColumn("source_file", input_file_name())
    
#     # Показываем пример данных
#     print("\n📄 Пример данных (первые 2 строки):")
#     df_with_source.select("source_file", "metadata").show(2, truncate=False)
    
#     # Получаем общее количество
#     total_records = df_with_source.count()
#     print(f"\n📁 Всего записей: {total_records}")
    
#     if total_records == 0:
#         print("❌ Нет данных для обработки")
#         return 0
    
#     # Получаем хеши уже скачанных
#     downloaded_hashes = get_downloaded_hashes()
    
#     # Создаём batch ID
#     current_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Собираем данные
#     print("📥 Загрузка данных в память...")
#     rows = df_with_source.collect()
    
#     new_images = []
#     downloaded_info = []
#     skipped_count = 0
#     error_count = 0
#     duplicate_in_batch = set()
    
#     print(f"\n🔍 Обработка {len(rows)} записей...")
    
#     for idx, row in enumerate(rows):
#         if idx % 10 == 0:
#             print(f"   Прогресс: {idx}/{len(rows)}")
        
#         try:
#             image_bytes = row["image"]
#             mask_bytes = row["mask_png"]
#             source_file = row["source_file"]
            
#             if image_bytes is None or mask_bytes is None:
#                 error_count += 1
#                 continue
            
#             # Вычисляем хеш
#             image_hash = calculate_hash(image_bytes)
            
#             # Проверяем дубликаты
#             if image_hash in downloaded_hashes or image_hash in duplicate_in_batch:
#                 skipped_count += 1
#                 continue
            
#             duplicate_in_batch.add(image_hash)
            
#             # Генерируем имя файла
#             filename = f"marked_{image_hash[:12]}.jpg"
#             local_file_path = os.path.join(LOCAL_PATH, filename)
            
#             # Проверяем существование
#             if os.path.exists(local_file_path):
#                 downloaded_hashes.add(image_hash)
#                 skipped_count += 1
#                 continue
            
#             new_images.append({
#                 'image_bytes': image_bytes,
#                 'mask_bytes': mask_bytes,
#                 'path': local_file_path,
#                 'hash': image_hash,
#                 'source_file': source_file,
#             })
            
#         except Exception as e:
#             print(f"⚠️ Ошибка обработки строки {idx}: {e}")
#             error_count += 1
    
#     print(f"\n📊 РЕЗУЛЬТАТЫ:")
#     print(f"   Всего: {len(rows)}")
#     print(f"   Пропущено: {skipped_count}")
#     print(f"   Ошибок: {error_count}")
#     print(f"   🆕 Новых: {len(new_images)}")
    
#     # Сохраняем новые изображения
#     if new_images:
#         print(f"\n🖼️ Сохранение {len(new_images)} изображений...")
        
#         for img_idx, img_info in enumerate(new_images):
#             try:
#                 if img_idx % 5 == 0:
#                     print(f"   Прогресс: {img_idx}/{len(new_images)}")
                
#                 # Накладываем маску
#                 overlay_bytes = overlay_mask_on_image(
#                     img_info['image_bytes'],
#                     img_info['mask_bytes']
#                 )
                
#                 if overlay_bytes is None:
#                     error_count += 1
#                     continue
                
#                 # Сохраняем
#                 with open(img_info['path'], "wb") as f:
#                     f.write(overlay_bytes)
                
#                 downloaded_info.append({
#                     'hash': img_info['hash'],
#                     'local_path': img_info['path'],
#                     'source_file': img_info['source_file'],
#                     'downloaded_at': datetime.now(),
#                     'size': str(len(overlay_bytes)),
#                     'batch_id': current_batch_id
#                 })
                
#             except Exception as e:
#                 print(f"❌ Ошибка сохранения: {e}")
#                 error_count += 1
        
#         # Сохраняем tracking
#         if downloaded_info:
#             save_tracking_info(downloaded_info)
#             save_batch_id(current_batch_id)
        
#         print(f"\n✅ Сохранено {len(downloaded_info)} изображений")
    
#     return len(downloaded_info)


# def main():
#     """Основная функция"""
#     print("🚀 ЗАПУСК ЗАГРУЗКИ РАЗМЕЧЕННЫХ ИЗОБРАЖЕНИЙ")
#     print("="*60)
    
#     # Создаём локальную директорию
#     os.makedirs(LOCAL_PATH, exist_ok=True)
    
#     try:
#         # Диагностика и получение DataFrame
#         df = diagnose_minio_access()
        
#         if df is not None:
#             # Обрабатываем данные
#             downloaded_count = process_dataframe(df)
            
#             if downloaded_count > 0:
#                 print(f"\n🎉 Успешно загружено {downloaded_count} изображений!")
#             else:
#                 print("\n✨ Нет новых изображений для загрузки")
#         else:
#             print("\n❌ Не удалось получить доступ к данным в MinIO")
#             print("\n💡 РЕКОМЕНДАЦИИ:")
#             print("1. Проверьте, что в bucket 'automarkup' есть parquet файлы")
#             print("2. Убедитесь, что контейнер с авторазметкой создал данные")
#             print("3. Проверьте права доступа к MinIO")
            
#             # Показываем статистику MinIO через mc если доступно
#             print("\n📊 Проверка через MinIO Client:")
#             os.system("mc ls local/automarkup/ 2>/dev/null || echo 'mc не установлен'")
            
#     except Exception as e:
#         print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
#         import traceback
#         traceback.print_exc()
#         raise
#     finally:
#         spark.stop()


# if __name__ == "__main__":
#     main()
