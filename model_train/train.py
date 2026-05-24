import os
import json
import zipfile
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import random_split

from transformers import (
    BeitForSemanticSegmentation,
    SegformerForSemanticSegmentation,
    Mask2FormerForUniversalSegmentation
)

# ============ КОНФИГУРАЦИЯ ============
DATA_DIR = "/workspace/data"
OUTPUT_DIR = "/workspace/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

num_epochs = 30
lr = 1e-4
batch_size = 2
num_classes = 11
img_size = (256, 256)


# ============ ДАТАСЕТ ============
class LaRSPanopticDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(512,512)):
        self.split_dir = os.path.join(root_dir, split)
        self.images_dir = os.path.join(self.split_dir, "images")
        self.masks_dir = os.path.join(self.split_dir, "panoptic_masks")
        self.img_size = img_size

        list_path = os.path.join(self.split_dir, "image_list.txt")
        with open(list_path, "r") as f:
            self.files = [x.strip() for x in f.readlines()]

        ann_path = os.path.join(self.split_dir, "panoptic_annotations.json")
        with open(ann_path, "r") as f:
            self.ann = json.load(f)

        self.file_to_mask = {x["file_name"]: x["id"] for x in self.ann["images"]}
        self.cat_ids = sorted([c["id"] for c in self.ann["categories"]])
        self.id2idx = {cid: i for i, cid in enumerate(self.cat_ids)}

    def __len__(self):
        return len(self.files)

    def _find_file(self, folder, name):
        for ext in [".png", ".jpg", ".jpeg"]:
            path = os.path.join(folder, name + ext)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No file found for {name} in {folder}")

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = self._find_file(self.images_dir, name)
        image = Image.open(img_path).convert("RGB")
        image = TF.resize(image, self.img_size)
        image = TF.to_tensor(image)

        mask_path = self._find_file(self.masks_dir, name)
        mask = Image.open(mask_path)
        mask = TF.resize(mask, self.img_size, interpolation=Image.NEAREST)
        mask = np.array(mask)

        if mask.ndim == 3:
            mask = mask[...,0]

        mask_idx = np.zeros_like(mask, dtype=np.int64)
        for k,v in self.id2idx.items():
            mask_idx[mask==k] = v

        mask_idx = torch.from_numpy(mask_idx)

        return image, mask_idx

# ============ МЕТРИКИ ============
def compute_mIoU(pred, target, num_classes, ignore_index=255):
    pred = pred.argmax(1)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    for cls in range(num_classes):
        pred_i = pred == cls
        target_i = target == cls

        mask = target != ignore_index
        intersection = np.logical_and(pred_i, target_i) & mask
        union = np.logical_or(pred_i, target_i) & mask

        if union.sum() == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection.sum() / union.sum())

    return np.nanmean(ious)

# ============ ОБЁРТКИ МОДЕЛЕЙ ============
class BEiT3Wrapper(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        
        # BEiT ожидает изображения 224x224 с патчами 16x16
        self.expected_size = 224
        self.patch_size = 16
        
        if pretrained:
            self.model = BeitForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                out_indices=[3, 5, 7, 11],  # Обязательно 4 индекса!
                image_size=self.expected_size,
                patch_size=self.patch_size
            )
        else:
            from transformers import BeitConfig
            config = BeitConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            config.out_indices = [3, 5, 7, 11]  # Обязательно 4 индекса!
            config.image_size = self.expected_size
            config.patch_size = self.patch_size
            self.model = BeitForSemanticSegmentation(config)
        
        self.classifier = nn.Identity()

    def forward(self, x):
        original_size = x.shape[-2:]
        
        # Масштабируем вход до 224x224
        x_resized = nn.functional.interpolate(
            x, 
            size=(self.expected_size, self.expected_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Пропускаем через модель
        outputs = self.model(pixel_values=x_resized)
        logits = outputs.logits
        
        # Масштабируем выход обратно до оригинального размера
        logits = nn.functional.interpolate(
            logits, 
            size=original_size, 
            mode="bilinear", 
            align_corners=False
        )
        
        return {"out": logits}


class SegFormerWrapper(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)
        self.classifier = nn.Identity()

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"out": logits}


class Mask2FormerWrapper(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        if pretrained:
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import Mask2FormerConfig
            config = Mask2FormerConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            self.model = Mask2FormerForUniversalSegmentation(config)
        self.num_classes = num_classes
        self.classifier = nn.Identity()

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        masks = outputs.masks_queries_logits
        classes = outputs.class_queries_logits
        classes = classes[:, :, :-1]
        masks_sigmoid = masks.sigmoid()
        classes_softmax = classes.softmax(dim=-1)
        logits = torch.einsum('bnij,bnc->bcij', masks_sigmoid, classes_softmax)
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return {"out": logits}


def train_model(model, model_name, train_loader, val_loader):
    """
    Обучение модели с агрессивной регуляризацией против переобучения
    """
    model = model.to(device)

    # Label smoothing для борьбы с переобучением
    criterion = nn.CrossEntropyLoss(ignore_index=255, label_smoothing=0.1)

    # AdamW с более сильной L2 регуляризацией
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )

    # Cosine Annealing с более частыми перезапусками
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )

    # Stochastic Weight Averaging для стабилизации
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = 5
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=1e-5)

    # Exponential Moving Average для сглаживания весов
    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                     avg_fn=lambda averaged_model_parameter,
                                                     model_parameter, num_averaged:
                                                     0.999 * averaged_model_parameter + 0.001 * model_parameter)

    # Параметры ранней остановки
    patience = 5
    early_stopping_counter = 0
    best_miou = 0.0
    best_val_loss = float('inf')

    history = {
        'train_loss': [],
        'train_miou': [],
        'val_loss': [],
        'val_miou': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        print(f"\n[{model_name}] Epoch {epoch+1}/{num_epochs}")

        # Динамический learning rate с warmup в начале
        if epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * 0.1
            print(f"📊 Warmup Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        else:
            print(f"📊 Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # TRAIN с агрессивной регуляризацией
        model.train()
        train_losses = []
        train_mious = []

        # Растущая вероятность MixUp (от 20% до 50%)
        mixup_prob = min(0.2 + epoch * 0.01, 0.5)

        for batch_idx, (imgs, masks) in enumerate(tqdm(train_loader, desc="Training")):
            imgs = imgs.to(device)
            masks = masks.to(device).long()

            # Пропускаем батчи размером 1 (проблема с BatchNorm)
            if imgs.size(0) < 2:
                continue

            optimizer.zero_grad()

            # Расширенная аугментация только если размер батча >= 2
            use_mixup = np.random.random() < mixup_prob and imgs.size(0) >= 2

            if use_mixup:
                # MixUp с увеличенной альфой
                alpha = 0.4
                lam = np.random.beta(alpha, alpha)

                batch_size = imgs.size(0)
                index = torch.randperm(batch_size).to(device)

                mixed_imgs = lam * imgs + (1 - lam) * imgs[index]

                outputs = model(mixed_imgs)["out"]

                # MixUp loss + label smoothing
                loss = lam * criterion(outputs, masks) + (1 - lam) * criterion(outputs, masks[index])

                # Дополнительная CutMix аугментация (20% случаев)
                if np.random.random() < 0.2 and batch_size >= 2:
                    # Случайный регион для CutMix
                    _, _, h, w = imgs.shape
                    cut_rat = np.random.uniform(0.3, 0.7)
                    cut_h, cut_w = int(h * cut_rat), int(w * cut_rat)
                    cy, cx = np.random.randint(0, h - cut_h), np.random.randint(0, w - cut_w)

                    # Применяем CutMix к части батча
                    mixed_imgs[:, :, cy:cy+cut_h, cx:cx+cut_w] = imgs[index][:, :, cy:cy+cut_h, cx:cx+cut_w]

                    outputs = model(mixed_imgs)["out"]

                    # Комбинированный loss для CutMix региона
                    mask_mix = masks.clone()
                    mask_mix[:, cy:cy+cut_h, cx:cx+cut_w] = masks[index][:, cy:cy+cut_h, cx:cx+cut_w]
                    loss = criterion(outputs, mask_mix)
            else:
                outputs = model(imgs)["out"]
                loss = criterion(outputs, masks)

            # Добавляем небольшой шум к градиентам для регуляризации
            if epoch > 2:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data += torch.randn_like(param.grad) * 1e-7

            loss.backward()

            # Более агрессивный gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()

            # Обновляем EMA модель
            ema_model.update_parameters(model)

            miou = compute_mIoU(outputs, masks, num_classes)
            train_losses.append(loss.item())
            train_mious.append(miou)

        # Проверяем, что есть данные для усреднения
        if len(train_losses) == 0:
            print("⚠️  Warning: No valid batches in this epoch!")
            continue

        avg_train_loss = np.mean(train_losses)
        avg_train_miou = np.mean(train_mious)
        print(f"Train | Loss: {avg_train_loss:.4f}, mIoU: {avg_train_miou:.4f}")

        # VAL с использованием EMA модели для лучшей оценки
        model.eval()
        val_losses = []
        val_mious = []

        # Используем EMA модель для валидации
        ema_model.eval()

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                masks = masks.to(device).long()

                # Пропускаем батчи размером 1
                if imgs.size(0) < 2:
                    continue

                # Валидация на EMA модели
                outputs = ema_model(imgs)["out"]

                val_loss = criterion(outputs, masks)
                miou = compute_mIoU(outputs, masks, num_classes)

                val_losses.append(val_loss.item())
                val_mious.append(miou)

        # Проверяем, что есть данные для усреднения
        if len(val_losses) == 0:
            print("⚠️  Warning: No valid validation batches!")
            continue

        avg_val_loss = np.mean(val_losses)
        avg_val_miou = np.mean(val_mious)
        print(f"Val   | Loss: {avg_val_loss:.4f}, mIoU: {avg_val_miou:.4f} (EMA)")

        # Сохраняем историю
        history['train_loss'].append(avg_train_loss)
        history['train_miou'].append(avg_train_miou)
        history['val_loss'].append(avg_val_loss)
        history['val_miou'].append(avg_val_miou)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Обновляем планировщик
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # SAVE BEST с учетом как mIoU, так и валидационного loss
        is_better = (avg_val_miou > best_miou) or \
                   (abs(avg_val_miou - best_miou) < 0.001 and avg_val_loss < best_val_loss)

        if is_better:
            best_miou = avg_val_miou
            best_val_loss = avg_val_loss
            early_stopping_counter = 0

            # Сохраняем лучшую модель
            save_path = os.path.join(OUTPUT_DIR, f"{model_name}_best.pth")
            torch.save(ema_model.state_dict(), save_path)

            # Также сохраняем SWA модель если доступна
            if epoch >= swa_start:
                swa_path = os.path.join(OUTPUT_DIR, f"{model_name}_swa.pth")
                torch.save(swa_model.state_dict(), swa_path)

            print(f"🔥 Best model saved: {save_path}")
            print(f"   mIoU: {avg_val_miou:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Полный чекпоинт
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': ema_model.state_dict(),
                'swa_state_dict': swa_model.state_dict() if epoch >= swa_start else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_miou': best_miou,
                'best_val_loss': best_val_loss,
                'history': history,
            }
            checkpoint_path = os.path.join(OUTPUT_DIR, f"{model_name}_checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)

            # Сохраняем историю
            np.save(os.path.join(OUTPUT_DIR, f"{model_name}_history.npy"), history)

        else:
            early_stopping_counter += 1
            print(f"⏱️  No improvement for {early_stopping_counter} epochs (patience: {patience})")

            # Ранняя остановка
            if early_stopping_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                print(f"   Best mIoU: {best_miou:.4f}, Best Val Loss: {best_val_loss:.4f}")
                break

        # Мониторинг переобучения
        overfitting_gap = avg_train_miou - avg_val_miou
        loss_gap = avg_val_loss - avg_train_loss

        if overfitting_gap > 0.05:
            print(f"⚠️  Overfitting detected: mIoU gap = {overfitting_gap:.4f}")
            if loss_gap > 0.05:
                print(f"   Loss gap also high: {loss_gap:.4f}")

            # Автоматически увеличиваем dropout для SegFormer/BEiT
            _increase_dropout(model, increment=0.05, max_dropout=0.5)

    print(f"\n✅ {model_name} finished. Best mIoU: {best_miou:.4f}")

    # Финальная оценка с SWA моделью
    if epoch >= swa_start:
        print("📊 Evaluating final SWA model...")

    return history


def _increase_dropout(model, increment=0.05, max_dropout=0.5, verbose=True):
    """
    Увеличивает dropout в модели, поддерживая SegFormer, BEiT и другие архитектуры
    """
    changed_params = []
    
    # Для моделей transformers (SegFormer, BEiT, etc.)
    if hasattr(model, 'model') and hasattr(model.model, 'config'):
        config = model.model.config
        
        dropout_params = [
            'hidden_dropout_prob',
            'attention_probs_dropout_prob', 
            'classifier_dropout_prob',
            'drop_path_rate'
        ]
        
        for param_name in dropout_params:
            if hasattr(config, param_name):
                old_value = getattr(config, param_name)
                new_value = min(old_value + increment, max_dropout)
                setattr(config, param_name, new_value)
                changed_params.append(f"{param_name}: {old_value:.3f} -> {new_value:.3f}")
        
        # Обновляем все Dropout слои
        dropout_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                old_p = module.p
                module.p = min(old_p + increment, max_dropout)
                dropout_count += 1
                if verbose:
                    changed_params.append(f"{name} ({type(module).__name__}): {old_p:.3f} -> {module.p:.3f}")
        
        if verbose and changed_params:
            print(f"   📊 Dropout adjustment summary:")
            for param in changed_params[:10]:  # Показываем первые 10 изменений
                print(f"      • {param}")
            if dropout_count > 10:
                print(f"      • ... and {dropout_count - 10} more dropout layers updated")
            print(f"      • Total dropout layers updated: {dropout_count}")
    
    # Для обычных моделей с прямым dropout
    elif hasattr(model, 'dropout'):
        if isinstance(model.dropout, nn.Dropout):
            old_p = model.dropout.p
            model.dropout.p = min(old_p + increment, max_dropout)
            if verbose:
                print(f"   📊 Increased main dropout: {old_p:.3f} -> {model.dropout.p:.3f}")
    
    # Рекурсивно для оберток
    elif hasattr(model, 'module'):
        _increase_dropout(model.module, increment, max_dropout, verbose)
    
    return len(changed_params) > 0

def get_train_parts(train_dataset):
    total_size = len(train_dataset)
    # Три независимых случайных подвыборки по 60%
    part_size = int(total_size * 0.6)
    indices = list(range(total_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_part1 = torch.utils.data.Subset(train_dataset, indices[:part_size])
    np.random.seed(123)
    np.random.shuffle(indices)
    train_part2 = torch.utils.data.Subset(train_dataset, indices[:part_size])
    np.random.seed(777)
    np.random.shuffle(indices)
    train_part3 = torch.utils.data.Subset(train_dataset, indices[:part_size])
    # Проверка пересечений
    idx1 = set(train_part1.indices)
    idx2 = set(train_part2.indices)
    idx3 = set(train_part3.indices)
    print(f"📊 Размер выборок: {part_size} из {total_size}")
    print(f"   Пересечение 1-2: {len(idx1 & idx2)}")
    print(f"   Пересечение 2-3: {len(idx2 & idx3)}")
    print(f"   Пересечение 1-3: {len(idx1 & idx3)}")
    print(f"   Общие для всех: {len(idx1 & idx2 & idx3)}")
    return train_part1, train_part2, train_part3

# ============ MAIN ============
if __name__ == "__main__":
    if not os.path.exists(os.path.join(DATA_DIR, "train")):
        raise FileNotFoundError(
            f"❌ Данные не найдены в {DATA_DIR}\n"
            "Скачай датасет на хосте:\n"
            "  wget https://box.vicos.si/lars/lars_v1.0.0_images.zip\n"
            "  wget https://box.vicos.si/lars/lars_v1.0.0_annotations.zip\n"
            "  unzip lars_v1.0.0_images.zip -d data/\n"
            "  unzip lars_v1.0.0_annotations.zip -d data/"
        )
    print("✅ Данные найдены")
    
    # Загружаем датасет
    train_dataset = LaRSPanopticDataset(DATA_DIR, split="train", img_size=img_size)
    train_dataset_224 = LaRSPanopticDataset(DATA_DIR, split="train", img_size=(224, 224))
    val_dataset = LaRSPanopticDataset(DATA_DIR, split="val", img_size=img_size)
    val_dataset_224 = LaRSPanopticDataset(DATA_DIR, split="val", img_size=(224, 224))

    train_part1, _, _ = get_train_parts(train_dataset_224)
    _, train_part2, train_part3 = get_train_parts(train_dataset)

    train_loader1 = DataLoader(train_part1, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_loader2 = DataLoader(train_part2, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_loader3 = DataLoader(train_part3, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    model1 = BEiT3Wrapper("microsoft/beit-base-patch16-224-pt22k-ft22k", num_classes=num_classes)
    model2 = SegFormerWrapper("nvidia/mit-b5", num_classes=num_classes)
    model3 = Mask2FormerWrapper("facebook/mask2former-swin-base-coco-instance", num_classes=num_classes)

    print("\n" + "="*50)
    print("🤖 Обучение BEiT")
    print("="*50)
    train_model(model1, "BEiT_Base_224x224", train_loader1, val_loader)
    del model1
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("🤖 Обучение SegFormer")
    print("="*50)
    train_model(model2, "SegFormer_B5_256x256", train_loader2, val_loader)
    del model2
    torch.cuda.empty_cache()

    print("\n" + "="*50)
    print("🤖 Обучение Mask2Former")
    print("="*50)
    train_model(model3, "Mask2Former_SwinB_256x256", train_loader3, val_loader)
    del model3
    torch.cuda.empty_cache()  

    print("\n🎉 Обучение завершено!")