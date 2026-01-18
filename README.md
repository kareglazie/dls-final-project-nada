# StyleGAN-NADA: Text-Driven Adaptation of Generative Models

## О проекте

**StyleGAN-NADA** (Non-Adversarial Domain Adaptation) — это метод адаптации генеративных моделей (в частности, StyleGAN2) к новым доменам с использованием только текстовых описаний, без необходимости обучать на реальных изображениях целевого домена. Этот проект представляет собой учебную реализацию метода с фокусом на адаптации лиц людей в различные художественные стили.

## Идея метода

**Ключевая идея**: использовать семантическую информацию из предобученной CLIP модели для "направления" адаптации генератора из исходного домена (например, реалистичные лица) в целевой домен (например, аниме-персонажи), используя только текстовые промпты.

### Как это работает:
1. **Базовый генератор**: предобученный StyleGAN2 на изображениях лиц (FFHQ)
2. **Текстовые промпты**:
   - Source: "Photo"
   - Target: "anime character, Japanese anime style"
3. **CLIP модель**: создает семантическое пространство, где можно измерить направление между текстовыми описаниями
4. **Обучение**: настраиваем генератор так, чтобы направление между сгенерированными изображениями соответствовало направлению между текстовыми промптами

### Функция потерь

**Direction Loss** 
```
L_dir = 1 - cos(E(G(z_t)) - E(G(z_s)), E(T_target) - E(T_source))
```
где:
- `E()` — энкодер CLIP
- `G()` — генератор
- `z_s, z_t` — латентные векторы
- `T_source, T_target` — текстовые промпты

## Последовательность работы над проектом

### Этап 1: Подготовка окружения

#### Автоматическая установка (рекомендуется)

Ноутбуки `train_and_save.ipynb` и `inference.ipynb` автоматически:
- Клонируют репозиторий stylegan-nada при отсутствии
- Скачивают базовую модель FFHQ при отсутствии
- Настраивают окружение для работы на CPU или CUDA

Просто запустите ноутбуки - все зависимости установятся автоматически!

#### Ручная установка (опционально)

1. **Клонирование репозиториев**:
   ```bash
   git clone https://github.com/NVlabs/stylegan2-ada.git stylegan_ada
   git clone https://github.com/rinongal/stylegan-nada.git stylegan_nada
   ```
2. **Установка зависимостей**:
   ```bash
   pip install -r requirements.txt
   pip install ftfy regex tqdm imageio gdown
   pip install git+https://github.com/openai/CLIP.git
   ```
3. **Загрузка базовой модели**:
   ```bash
   mkdir -p models
   gdown --id 1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT -O models/ffhq.pt
   ```

### Поддержка CPU

Проект поддерживает работу как на GPU, так и на CPU без необходимости компиляции CUDA расширений.

**Как это работает:**
- При наличии CUDA: используются оптимизированные CUDA расширения
- Без CUDA: автоматически используются CPU-реализации операций (fused_act, upfirdn2d) без необходимости компиляции

**Автоматическое применение CPU-поддержки:**

В ноутбуке с инференсом автоматически вызывается скрипт для применения CPU-поддержки stylegan-nada:
- Скрипт `setup_cpu_support.py` проверяет устройство (CPU/GPU)
- Если устройство CPU, автоматически модифицирует файлы `fused_act.py` и `upfirdn2d.py`
- Создает резервные копии оригинальных файлов (`.backup`)

**Ручной запуск скрипта (опционально):**

Если нужно применить CPU-поддержку вручную:
```bash
python setup_cpu_support.py --stylegan-nada-dir ../stylegan_nada
```

### Этап 2: Обучение моделей и инференс

1. **Обучение** реализовано в блокноте `notebooks/train_and_save.ipynb`

2. **Инференс** (генерация случайных изображений) реализован в `notebooks/inference.ipynb`
   - клонирует репозиторий stylegan-nada при отсутствии
   - скачивает базовую модель FFHQ при отсутствии
   - применяет CPU-поддержку при отсутствии CUDA
   - результаты сохраняются в `notebooks/inference_output`

### Этап 3: Создание веб-приложения

Веб-приложение на Streamlit включает три раздела:
- **Генерация**: интерактивная генерация изображений в различных стилях с настройкой seed
- **Визуализации**: графики качества моделей, сравнение конфигураций, графики сходимости
- **Отчет**: детальные метрики всех экспериментов и статистика

## Реализованные стили

### 1. **Аниме-стиль**
```
Target prompt: "anime character, Japanese anime style, cel-shaded, vibrant colors, large expressive eyes"
```

### 2. **Скетч-стиль**
```
Target prompt: "sketch, pencil drawing, hand-drawn, monochrome, artistic sketch, black and white drawing"
```

### 3. **Джокер-стиль**
```
Target prompt: "the Joker from Batman, green hair, white face, red lips, creepy smile, makeup"
```

### 4. **Картина маслом**
```
Target prompt: "oil painting, brush strokes, artistic, textured, classical painting style, canvas texture"
```

## Результаты обучения

Всего обучено **12 моделей** (4 стиля × 3 конфигурации заморозки слоев).

### Лучшие результаты:
- **Аниме**: `anime_style_freeze_2` (cosine similarity: 0.1851)
- **Скетч**: `sketch_style_freeze_0` (cosine similarity: 0.1675)
- **Джокер**: `joker_style_freeze_2` (cosine similarity: 0.2241)
- **Картина маслом**: `oil_painting_style_freeze_2` (cosine similarity: 0.2085)

### Выводы:
- Заморозка 2 слоев показала лучшие результаты в среднем
- Стиль Джокера дал наилучшее качество адаптации
- Все модели успешно сохраняют структуру лица при стилизации

## Примеры генерации

Ниже представлены примеры изображений, сгенерированных обученными моделями. Все изображения были созданы с использованием базовой модели StyleGAN2 FFHQ, адаптированной к различным стилям.

### Аниме-стиль (anime_style_freeze_2)

Примеры генераций лучшей модели аниме-стиля:

| Итерация 0 | Итерация 300 | Итерация 600 |
|------------|--------------|--------------|
| ![Аниме 0](output/anime_style_freeze_2/sample/dst_000000.jpg) | ![Аниме 300](output/anime_style_freeze_2/sample/dst_000300.jpg) | ![Аниме 600](output/anime_style_freeze_2/sample/dst_000599.jpg) |

### Скетч-стиль (sketch_style_freeze_0)

Примеры генераций лучшей модели скетч-стиля:

| Итерация 0 | Итерация 300 | Итерация 600 |
|------------|--------------|--------------|
| ![Скетч 0](output/sketch_style_freeze_0/sample/dst_000000.jpg) | ![Скетч 300](output/sketch_style_freeze_0/sample/dst_000300.jpg) | ![Скетч 600](output/sketch_style_freeze_0/sample/dst_000599.jpg) |

### Стиль Джокера (joker_style_freeze_2)

Примеры генераций лучшей модели стиля Джокера:

| Итерация 0 | Итерация 300 | Итерация 600 |
|------------|--------------|--------------|
| ![Джокер 0](output/joker_style_freeze_2/sample/dst_000000.jpg) | ![Джокер 300](output/joker_style_freeze_2/sample/dst_000300.jpg) | ![Джокер 600](output/joker_style_freeze_2/sample/dst_000599.jpg) |

### Картина маслом (oil_painting_style_freeze_2)

Примеры генераций лучшей модели стиля картины маслом:

| Итерация 0 | Итерация 300 | Итерация 600 |
|------------|--------------|--------------|
| ![Картина 0](output/oil_painting_style_freeze_2/sample/dst_000000.jpg) | ![Картина 300](output/oil_painting_style_freeze_2/sample/dst_000300.jpg) | ![Картина 600](output/oil_painting_style_freeze_2/sample/dst_000599.jpg) |

### Сравнение конфигураций заморозки

Для каждого стиля были обучены модели с разным количеством замороженных слоев (0, 2, 4). Ниже представлены примеры для аниме-стиля:

| Freeze 0 | Freeze 2 | Freeze 4 |
|----------|----------|----------|
| ![Аниме freeze 0](output/anime_style_freeze_0/sample/dst_000300.jpg) | ![Аниме freeze 2](output/anime_style_freeze_2/sample/dst_000300.jpg) | ![Аниме freeze 4](output/anime_style_freeze_4/sample/dst_000300.jpg) |

## Технические детали

### Гиперпараметры обучения:
- **Learning rate**: 0.002
- **Training iterations**: 600
- **Batch size**: 2
- **Заморозка слоев**: 0, 2, 4
- **CLIP модели**: ансамбль из ViT-B/32, ViT-B/16, ViT-L/14
- **CLIP веса**: [0.4, 0.1, 0.5]

**Базовый генератор**: StyleGAN2 (1024×1024, FFHQ)

## Источники

### Основные статьи:
1. **StyleGAN-NADA**: [arXiv:2108.00946](https://arxiv.org/pdf/2108.00946)
2. **StyleGAN2**: [arXiv:1912.04958](https://arxiv.org/pdf/1912.04958)
3. **StyleCLIP**: [arXiv:2103.17249](https://arxiv.org/pdf/2103.17249)

### Репозитории:
1. [Official StyleGAN-NADA](https://github.com/rinongal/stylegan-nada)
2. [StyleGAN-NADA Demo](https://stylegan-nada.github.io/)
3. [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)

