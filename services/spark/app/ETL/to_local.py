from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BinaryType, TimestampType
from pyspark.sql.functions import input_file_name, md5, col
import os
import hashlib
from datetime import datetime

spark = SparkSession.builder.appName("MinIO-ToLocal-Incremental").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# --- MinIO config ---
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3a.endpoint", "http://minio:9000")
hadoop_conf.set("fs.s3a.access.key", "minioadmin")
hadoop_conf.set("fs.s3a.secret.key", "minioadmin")
hadoop_conf.set("fs.s3a.path.style.access", "true")
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")

# --- Пути ---
STAGED_PATH = "s3a://staged/images_parquet/"
LOCAL_PATH = "/data/filtered_photos/"
TRACKING_FILE = "/data/filtered_photos/.downloaded_tracking.parquet"
CHECKPOINT_FILE = "/data/filtered_photos/.last_batch.txt"

# --- Схемы ---
image_schema = StructType([StructField("image", BinaryType(), True)])

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
            downloaded_hashes = set(row.file_hash for row in tracking_df.select("file_hash").collect())
            print(f"📋 Найдено {len(downloaded_hashes)} ранее скачанных изображений")
            return downloaded_hashes
        except Exception as e:
            print(f"⚠️ Ошибка чтения tracking файла: {e}")
            return set()
    else:
        print("📋 Tracking файл не найден, начинаем загрузку с чистого листа")
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
    
    # Сохраняем tracking информацию
    tracking_df.write.mode("append").parquet(f"file://{TRACKING_FILE}")
    print(f"💾 Обновлен tracking файл: добавлено {len(downloaded_info)} записей")


def get_files_by_modification_time():
    """Получает список parquet файлов с их временем модификации через Spark"""
    try:
        # Используем Spark SQL для получения списка файлов с временем модификации
        files_df = spark.sql(f"SELECT * FROM parquet.`{STAGED_PATH}`")
        
        # Получаем список всех parquet файлов в директории
        file_list = []
        fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
            spark._jsc.hadoopConfiguration()
        )
        
        # Конвертируем URI в Path
        path_uri = spark._jvm.java.net.URI(STAGED_PATH)
        path = spark._jvm.org.apache.hadoop.fs.Path(path_uri)
        
        # Получаем статусы файлов
        file_statuses = fs.listStatus(path)
        
        for status in file_statuses:
            file_path = str(status.getPath())
            if file_path.endswith(".parquet"):
                file_list.append({
                    'path': file_path,
                    'modification_time': status.getModificationTime(),
                    'size': status.getLen()
                })
        
        return file_list
    except Exception as e:
        print(f"⚠️ Не удалось получить список файлов: {e}")
        return []


def download_new_images():
    """Скачивает только новые изображения из staged"""
    
    # Создаём директорию, если её нет
    os.makedirs(LOCAL_PATH, exist_ok=True)
    
    # Получаем хеши уже скачанных изображений
    downloaded_hashes = get_downloaded_hashes()
    
    # Получаем ID последнего батча
    last_batch_id = get_last_batch_id()
    
    # Создаём новый batch ID
    current_batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Проверяем, не был ли этот батч уже обработан
    if last_batch_id:
        print(f"📌 Последний обработанный батч: {last_batch_id}")
    
    # Читаем все изображения из staged
    print("📖 Чтение изображений из staged...")
    
    try:
        # Пытаемся получить список файлов для информации
        file_list = get_files_by_modification_time()
        if file_list:
            print(f"📁 Найдено parquet файлов: {len(file_list)}")
            # Можно добавить логику фильтрации по времени, если нужно
    except Exception as e:
        print(f"⚠️ Не удалось проанализировать файлы: {e}")
    
    # Читаем все изображения (это неизбежно, но мы фильтруем по хешу)
    df = spark.read.schema(image_schema).format("parquet").load(STAGED_PATH)
    
    # Добавляем информацию о исходном файле
    df_with_source = df.withColumn("source_file", input_file_name())
    
    # Собираем все строки
    rows = df_with_source.collect()
    total_images = len(rows)
    print(f"📁 Всего изображений в staged: {total_images}")
    
    # Фильтруем новые изображения
    new_images = []
    downloaded_info = []
    skipped_count = 0
    duplicate_in_batch = set()  # Для отслеживания дубликатов внутри одного батча
    
    print("🔍 Проверка новых изображений...")
    
    for idx, row in enumerate(rows):
        image_bytes = row["image"]
        source_file = row["source_file"]
        
        if image_bytes is None:
            continue
            
        # Вычисляем хеш изображения
        image_hash = calculate_hash(image_bytes)
        
        # Пропускаем, если уже скачано ранее
        if image_hash in downloaded_hashes:
            skipped_count += 1
            continue
        
        # Пропускаем дубликаты внутри текущего батча
        if image_hash in duplicate_in_batch:
            print(f"⚠️ Обнаружен дубликат в staged: {image_hash[:12]}")
            continue
        
        duplicate_in_batch.add(image_hash)
        
        # Генерируем имя файла на основе хеша (без timestamp, чтобы избежать дублирования)
        filename = f"img_{image_hash[:12]}.jpg"
        local_file_path = os.path.join(LOCAL_PATH, filename)
        
        # Проверяем, не существует ли уже такой файл
        if os.path.exists(local_file_path):
            print(f"⚠️ Файл уже существует локально: {filename}")
            # Добавляем хеш в downloaded_hashes, чтобы не обрабатывать повторно
            downloaded_hashes.add(image_hash)
            skipped_count += 1
            continue
        
        new_images.append({
            'bytes': image_bytes,
            'path': local_file_path,
            'hash': image_hash,
            'source_file': source_file,
            'size': len(image_bytes)
        })
    
    print(f"🆕 Новых изображений для скачивания: {len(new_images)}")
    print(f"⏭️ Пропущено (уже скачано): {skipped_count}")
    
    # Скачиваем новые изображения
    if new_images:
        print(f"💾 Сохранение {len(new_images)} новых изображений...")
        
        for img_info in new_images:
            try:
                # Сохраняем файл
                with open(img_info['path'], "wb") as f:
                    f.write(img_info['bytes'])
                
                # Добавляем информацию для tracking
                downloaded_info.append({
                    'hash': img_info['hash'],
                    'local_path': img_info['path'],
                    'source_file': img_info['source_file'],
                    'downloaded_at': datetime.now(),
                    'size': str(img_info['size']),
                    'batch_id': current_batch_id
                })
                
            except Exception as e:
                print(f"❌ Ошибка при сохранении {img_info['path']}: {e}")
        
        # Сохраняем tracking информацию
        if downloaded_info:
            save_tracking_info(downloaded_info)
            # Сохраняем ID батча только если были реальные загрузки
            save_batch_id(current_batch_id)
            
        print(f"✅ Успешно сохранено {len(downloaded_info)} новых изображений")
    else:
        print("✨ Нет новых изображений для скачивания")
        # Если нет новых изображений, не обновляем batch_id
    
    # Выводим статистику
    print("\n" + "="*50)
    print("📊 Статистика загрузки:")
    print(f"   Всего изображений в staged: {total_images}")
    print(f"   Уже скачано ранее: {len(downloaded_hashes)}")
    print(f"   Новых скачано: {len(downloaded_info)}")
    print(f"   Пропущено в этой сессии: {skipped_count}")
    print(f"   Batch ID: {current_batch_id}")
    print(f"   Локальная директория: {LOCAL_PATH}")
    print("="*50)
    
    return len(downloaded_info)


def main():
    """Основная функция"""
    print("🚀 Запуск инкрементальной загрузки изображений из staged на локальный диск")
    print(f"📂 Исходная директория: {STAGED_PATH}")
    print(f"📂 Локальная директория: {LOCAL_PATH}")
    print(f"📋 Tracking файл: {TRACKING_FILE}")
    
    try:
        downloaded_count = download_new_images()
        
        if downloaded_count > 0:
            print(f"🎉 Загрузка завершена успешно! Скачано {downloaded_count} новых изображений")
        else:
            print("🎉 Проверка завершена. Новых изображений нет.")
            
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main()