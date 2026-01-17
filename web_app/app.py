import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import json
import pandas as pd
import gdown
from pathlib import Path
import subprocess
import shutil

st.set_page_config(
    page_title="StyleGAN-NADA Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

page = st.sidebar.selectbox(
    "Выберите раздел", ["🎨 Генерация", "📊 Визуализации", "📋 Отчет"]
)

BEST_MODELS = {
    "anime_style": {
        "model_key": "anime_style_freeze_2",
        "name": "Аниме-стиль",
        "cos_sim": 0.1851,
    },
    "sketch_style": {
        "model_key": "sketch_style_freeze_0",
        "name": "Скетч-стиль",
        "cos_sim": 0.1675,
    },
    "joker_style": {
        "model_key": "joker_style_freeze_2",
        "name": "Стиль Джокера",
        "cos_sim": 0.2241,
    },
    "oil_painting_style": {
        "model_key": "oil_painting_style_freeze_2",
        "name": "Картина маслом",
        "cos_sim": 0.2085,
    },
}




@st.cache_resource
def download_stylegan_nada():
    stylegan_nada_dir = Path("stylegan_nada")

    if stylegan_nada_dir.exists() and (stylegan_nada_dir / "ZSSGAN").exists():
        return str(stylegan_nada_dir)

    with st.spinner("Клонирование репозитория stylegan-nada из GitHub..."):
        try:
            result = subprocess.run(
                ["git", "--version"], capture_output=True, text=True
            )
            if result.returncode != 0:
                st.error(
                    "Git не установлен! Установите Git для клонирования репозитория."
                )
                return None

            if stylegan_nada_dir.exists():
                shutil.rmtree(stylegan_nada_dir)

            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "https://github.com/rinongal/stylegan-nada.git",
                    str(stylegan_nada_dir),
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0 and (stylegan_nada_dir / "ZSSGAN").exists():
                return str(stylegan_nada_dir)
            else:
                st.error(f"Ошибка при клонировании репозитория: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            st.error("Таймаут при клонировании репозитория. Попробуйте позже.")
            return None
        except Exception as e:
            st.error(f"Ошибка при клонировании репозитория: {e}")
            return None


@st.cache_resource
def download_base_model():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "ffhq.pt"

    if model_path.exists():
        return str(model_path)

    with st.spinner("Загрузка базовой модели FFHQ из облака..."):
        try:
            gdown.download(
                "https://drive.google.com/uc?id=1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT",
                str(model_path),
                quiet=True,
            )
            if model_path.exists():
                return str(model_path)
            else:
                st.error("Не удалось загрузить базовую модель")
                return None
        except Exception as e:
            st.error(f"Ошибка при загрузке базовой модели: {e}")
            return None


@st.cache_resource
def load_models():
    device = "cpu"
    
    if "CUDA_HOME" not in os.environ:
        import tempfile
        temp_cuda = tempfile.mkdtemp()
        os.environ["CUDA_HOME"] = temp_cuda
        os.makedirs(os.path.join(temp_cuda, "lib64"), exist_ok=True)
        os.makedirs(os.path.join(temp_cuda, "lib"), exist_ok=True)

    base_model_path = download_base_model()
    if not base_model_path:
        return None

    stylegan_nada_dir = download_stylegan_nada()
    if not stylegan_nada_dir:
        return None

    sys.path.append(stylegan_nada_dir)
    sys.path.append(os.path.join(stylegan_nada_dir, "ZSSGAN"))

    try:
        from ZSSGAN.model.ZSSGAN import SG2Generator
    except ImportError as e:
        st.error(f"Ошибка импорта ZSSGAN: {e}")
        st.info("Убедитесь, что репозиторий stylegan-nada клонирован правильно")
        return None

    models = {}
    models_dir = "models"

    for style_key, style_info in BEST_MODELS.items():
        model_key = style_info["model_key"]
        model_file = f"final_model_{model_key}.pt"
        model_path = os.path.join(models_dir, model_file)

        if not os.path.exists(model_path):
            st.warning(f"Модель {model_key} не найдена по пути: {model_path}")
            continue

        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            metadata = checkpoint.get("metadata", {})

            generator_wrapper = SG2Generator(
                base_model_path, img_size=1024, device=device
            )

            if "generator_state_dict" in checkpoint:
                state_dict = checkpoint["generator_state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("generator."):
                        new_key = key[len("generator.") :]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value

                generator_wrapper.generator.load_state_dict(
                    new_state_dict, strict=False
                )

            generator_wrapper.eval()

            models[style_key] = {
                "generator": generator_wrapper,
                "name": style_info["name"],
                "model_key": model_key,
                "cos_sim": style_info["cos_sim"],
                "metadata": metadata,
            }

        except Exception as e:
            st.warning(f"Не удалось загрузить {model_key}: {e}")

    if not models:
        st.error("❌ Не найдено обученных моделей!")
        return None

    return models


def tensor_to_pil(tensor):
    img_np = (
        ((tensor.permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Image.fromarray(img_np)


if page == "🎨 Генерация":
    st.title("🎨 StyleGAN-NADA Generator")
    st.markdown("### Генерация изображений в различных стилях")

    with st.spinner("Загрузка моделей..."):
        models = load_models()

    if not models:
        st.error("❌ Модели не загружены!")
        st.stop()

    with st.sidebar:
        st.header("⚙️ Настройки генерации")

        st.subheader("🎨 Выберите стиль:")
        style_options = {info["name"]: key for key, info in models.items()}
        selected_style_name = st.selectbox("Стиль:", list(style_options.keys()))
        selected_style_key = style_options[selected_style_name]

        selected_model = models[selected_style_key]
        st.info(f"**Модель:** {selected_model['name']}")
        st.info(f"**Качество:** Cosine similarity: {selected_model['cos_sim']:.4f}")

        st.subheader("🎲 Параметры генерации:")

        seed_mode = st.radio("Режим seed:", ["🎲 Случайный", "🔢 Задать свой"])

        if seed_mode == "🎲 Случайный":
            if st.button("🎲 Новый случайный seed"):
                import random

                seed = random.randint(0, 1000000)
                st.session_state.seed = seed
            else:
                seed = st.session_state.get("seed", 42)
        else:
            seed = st.number_input(
                "Seed:",
                value=st.session_state.get("seed", 42),
                min_value=0,
                max_value=1000000,
                step=1,
            )
            st.session_state.seed = seed

        truncation = st.slider(
            "Truncation:",
            0.1,
            1.0,
            0.7,
            0.1,
            help="Контролирует разнообразие генераций. Меньше = более реалистично, больше = более разнообразно",
        )

        num_images = st.number_input(
            "Количество изображений:", min_value=1, max_value=4, value=1, step=1
        )

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🎨 Сгенерировать!", type="primary", width="stretch"):
            with st.spinner("Генерация изображений..."):
                try:
                    generator = selected_model["generator"]

                    torch.manual_seed(seed)
                    images = []

                    with torch.no_grad():
                        for i in range(num_images):
                            z = torch.randn(1, 512)

                            truncation_latent = None
                            if truncation < 1.0:
                                if hasattr(generator, "mean_latent"):
                                    if isinstance(generator.mean_latent, torch.Tensor):
                                        truncation_latent = generator.mean_latent
                                    else:
                                        truncation_latent = generator.mean_latent(
                                            n_latent=2048
                                        )
                                else:
                                    w_avg_samples = []
                                    for _ in range(100):
                                        z_sample = torch.randn(1, 512)
                                        if hasattr(generator, "generator"):
                                            w_sample = generator.generator.style(
                                                z_sample
                                            )
                                        else:
                                            w_sample = generator.style(z_sample)
                                        w_avg_samples.append(w_sample)
                                    truncation_latent = torch.cat(
                                        w_avg_samples, dim=0
                                    ).mean(dim=0, keepdim=True)

                            image, _ = generator(
                                [z],
                                truncation=truncation,
                                truncation_latent=truncation_latent,
                            )
                            images.append(image[0])

                    if num_images == 1:
                        img_pil = tensor_to_pil(images[0])
                        st.image(
                            img_pil,
                            caption=f"Стиль: {selected_model['name']} | Seed: {seed}",
                            width="stretch",
                        )

                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        st.download_button(
                            label="📥 Скачать изображение",
                            data=buf.getvalue(),
                            file_name=f"{selected_style_key}_seed{seed}.png",
                            mime="image/png",
                            width="stretch",
                        )
                    else:
                        cols = st.columns(num_images)
                        for idx, img_tensor in enumerate(images):
                            with cols[idx]:
                                img_pil = tensor_to_pil(img_tensor)
                                st.image(
                                    img_pil,
                                    caption=f"Seed: {seed + idx}",
                                    width="stretch",
                                )

                                buf = io.BytesIO()
                                img_pil.save(buf, format="PNG")
                                st.download_button(
                                    label="📥 Скачать",
                                    data=buf.getvalue(),
                                    file_name=f"{selected_style_key}_seed{seed + idx}.png",
                                    mime="image/png",
                                    key=f"download_{idx}",
                                )

                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")
                    import traceback

                    st.code(traceback.format_exc())

    with col2:
        st.info(f"**Стиль:** {selected_model['name']}")
        st.info(f"**Качество:** {selected_model['cos_sim']:.4f}")

        metadata = selected_model.get("metadata", {})
        if metadata:
            target_class = metadata.get("target_class", "N/A")
            st.info(f"**Промпт:** {target_class[:60]}...")

        st.info(f"**Seed:** {seed}")
        st.info(f"**Truncation:** {truncation}")
        st.info(f"**Количество:** {num_images}")

elif page == "📊 Визуализации":
    st.title("📊 Визуализации результатов обучения")

    output_dir = "output"
    visualizations_dir = os.path.join(output_dir, "visualizations")

    if not os.path.exists(visualizations_dir):
        st.warning("Директория с визуализациями не найдена!")
        st.info("Запустите ноутбук train_and_save.ipynb для создания визуализаций")
    else:
        st.header("📈 Графики качества моделей")

        col1, col2 = st.columns(2)

        with col1:
            quality_by_style = os.path.join(visualizations_dir, "quality_by_style.png")
            if os.path.exists(quality_by_style):
                st.subheader("Качество по стилям")
                st.image(quality_by_style, width="stretch")

            effect_of_freeze = os.path.join(visualizations_dir, "effect_of_freeze.png")
            if os.path.exists(effect_of_freeze):
                st.subheader("Влияние заморозки слоев")
                st.image(effect_of_freeze, width="stretch")

        with col2:
            quality_heatmap = os.path.join(visualizations_dir, "quality_heatmap.png")
            if os.path.exists(quality_heatmap):
                st.subheader("Heatmap качества")
                st.image(quality_heatmap, width="stretch")

            convergence_best = os.path.join(
                visualizations_dir, "convergence_best_models.png"
            )
            if os.path.exists(convergence_best):
                st.subheader("Сходимость лучших моделей")
                st.image(convergence_best, width="stretch")

        st.header("🎨 Сравнение конфигураций по стилям")

        comparison_files = {
            "Аниме": "comparison_anime_style.png",
            "Джокер": "comparison_joker_style.png",
            "Скетч": "comparison_sketch_style.png",
            "Картина маслом": "comparison_oil_painting_style.png",
        }

        cols = st.columns(2)
        for idx, (style_name, filename) in enumerate(comparison_files.items()):
            filepath = os.path.join(visualizations_dir, filename)
            if os.path.exists(filepath):
                with cols[idx % 2]:
                    st.subheader(style_name)
                    st.image(filepath, width="stretch")

        st.header("📉 Графики сходимости")
        convergence_dir = os.path.join(visualizations_dir, "convergence_plots")

        if os.path.exists(convergence_dir):
            convergence_files = [
                f for f in os.listdir(convergence_dir) if f.endswith(".png")
            ]
            if convergence_files:
                cols = st.columns(2)
                for idx, filename in enumerate(convergence_files[:4]):
                    filepath = os.path.join(convergence_dir, filename)
                    with cols[idx % 2]:
                        st.image(filepath, width="stretch", caption=filename)

elif page == "📋 Отчет":
    st.title("📋 Отчет о проекте")

    st.header("📊 Метрики моделей")

    metrics_file = "output/detailed_metrics_all_experiments.json"

    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

        st.subheader("Лучшие модели по стилям")

        best_models_table = []
        for style_key, style_info in BEST_MODELS.items():
            best_models_table.append(
                {
                    "Стиль": style_info["name"],
                    "Модель": style_info["model_key"],
                    "Cosine Similarity": f"{style_info['cos_sim']:.4f}",
                }
            )

        df_best = pd.DataFrame(best_models_table)
        st.dataframe(df_best, width="stretch", hide_index=True)

        st.subheader("Сводная таблица всех экспериментов")

        models_data = metrics_data.get("models", {})
        if models_data:
            rows = []
            for model_name, model_info in models_data.items():
                if model_info.get("exists"):
                    rows.append(
                        {
                            "Модель": model_name,
                            "Стиль": model_info.get("base_model", "N/A"),
                            "Заморожено слоев": model_info.get("freeze_layers", "N/A"),
                            "Финальный Loss": (
                                f"{model_info.get('final_loss', 0):.4f}"
                                if model_info.get("final_loss")
                                else "N/A"
                            ),
                            "Cosine Similarity": (
                                f"{model_info.get('cos_sim', 0):.4f}"
                                if model_info.get("cos_sim")
                                else "N/A"
                            ),
                            "Размер (MB)": (
                                f"{model_info.get('size_mb', 0):.2f}"
                                if model_info.get("size_mb")
                                else "N/A"
                            ),
                        }
                    )

            if rows:
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)

        st.subheader("Статистика")
        stats = metrics_data.get("statistics", {})
        if stats:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Средний Loss",
                    (
                        f"{stats.get('avg_loss', 0):.4f}"
                        if stats.get("avg_loss")
                        else "N/A"
                    ),
                )
            with col2:
                st.metric(
                    "Средний Cosine Similarity",
                    (
                        f"{stats.get('avg_cos_sim', 0):.4f}"
                        if stats.get("avg_cos_sim")
                        else "N/A"
                    ),
                )
            with col3:
                st.metric(
                    "Общий размер моделей (MB)",
                    (
                        f"{stats.get('total_size_mb', 0):.2f}"
                        if stats.get("total_size_mb")
                        else "N/A"
                    ),
                )

        st.subheader("Лучшие модели по стилям")
        best_models = metrics_data.get("summary", {}).get("best_models_by_style", {})
        if best_models:
            for style, info in best_models.items():
                st.write(
                    f"**{style}**: {info.get('model_name', 'N/A')} (cos_sim: {info.get('cos_sim', 0):.4f})"
                )

    else:
        st.warning("Файл с метриками не найден!")
        st.info("Запустите ноутбук train_and_save.ipynb для создания метрик")

    st.header("ℹ️ Информация о проекте")

    st.markdown(
        """
    ### Описание метода
    
    **StyleGAN-NADA** (Non-Adversarial Domain Adaptation) — метод адаптации генеративных моделей 
    к новым доменам с использованием только текстовых описаний.
    
    ### Реализованные стили:
    
    1. **Аниме-стиль** - адаптация в стиль японской анимации
    2. **Скетч-стиль** - имитация карандашного рисунка
    3. **Стиль Джокера** - трансформация с характерными чертами персонажа
    4. **Картина маслом** - эффект масляной живописи
    
    ### Технические детали:
    
    - Базовая модель: StyleGAN2 FFHQ (1024×1024)
    - CLIP модели: ансамбль из ViT-B/32, ViT-B/16, ViT-L/14
    - Learning rate: 0.002
    - Итерации обучения: 600
    - Эксперименты с заморозкой слоев: 0, 2, 4
    
    ### Результаты:
    
    Всего обучено **12 моделей** (4 стиля × 3 конфигурации заморозки).
    В приложении используются лучшие модели по каждому стилю.
    """
    )
