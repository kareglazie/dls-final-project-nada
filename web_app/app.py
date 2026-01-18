import os
import sys
import logging
import traceback

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FORCE_CUDA"] = "0"

import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
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

    base_model_path = download_base_model()
    if not base_model_path:
        return None

    stylegan_nada_dir = download_stylegan_nada()
    if not stylegan_nada_dir:
        return None

    if device == "cpu":
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            setup_script = os.path.join(project_root, "setup_cpu_support.py")

            if not os.path.exists(setup_script):
                setup_script = os.path.join(os.getcwd(), "setup_cpu_support.py")

            if os.path.exists(setup_script):
                sys.path.insert(0, os.path.dirname(setup_script))
                from setup_cpu_support import setup_cpu_support

                setup_cpu_support(stylegan_nada_dir)
            else:
                st.warning(
                    f"Скрипт setup_cpu_support.py не найден. Искали: {setup_script}"
                )
                st.info(
                    "Попробуйте запустить скрипт вручную: python setup_cpu_support.py"
                )
        except Exception as e:
            st.warning(f"Не удалось применить CPU-поддержку: {e}")
            import traceback

            st.code(traceback.format_exc())

    sys.path.append(stylegan_nada_dir)
    sys.path.append(os.path.join(stylegan_nada_dir, "ZSSGAN"))

    try:
        from ZSSGAN.model.ZSSGAN import SG2Generator
    except ImportError as e:
        st.error(f"Ошибка импорта ZSSGAN: {e}")
        st.info("Убедитесь, что репозиторий stylegan-nada клонирован правильно")
        return None

    try:
        base_checkpoint = torch.load(base_model_path, map_location="cpu")
    except Exception as e:
        st.error(f"❌ Ошибка при предзагрузке базовой модели: {e}")
        import traceback

        with st.expander("🔍 Детали ошибки загрузки базовой модели"):
            st.code(traceback.format_exc())
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

            if hasattr(generator_wrapper, "generator"):
                generator_wrapper.generator = generator_wrapper.generator.to(device)
                for name, param in generator_wrapper.generator.named_parameters():
                    if param.device.type != "cpu":
                        st.warning(
                            f"⚠️ {style_key}: Параметр {name} базового генератора не на CPU: {param.device}"
                        )
                        generator_wrapper.generator = generator_wrapper.generator.to(
                            device
                        )
                        break

            if "generator_state_dict" in checkpoint:
                state_dict = checkpoint["generator_state_dict"]
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("generator."):
                        new_key = key[len("generator.") :]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value

                missing_keys, unexpected_keys = (
                    generator_wrapper.generator.load_state_dict(
                        new_state_dict, strict=False
                    )
                )

                if missing_keys:
                    st.warning(
                        f"⚠️ {style_key}: Отсутствующие ключи при загрузке ({len(missing_keys)}): {list(missing_keys)[:5]}..."
                    )
                if unexpected_keys:
                    st.warning(
                        f"⚠️ {style_key}: Неожиданные ключи при загрузке ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}..."
                    )

            generator_wrapper.eval()

            if hasattr(generator_wrapper, "generator"):
                generator_wrapper.generator = generator_wrapper.generator.to(device)
                generator_wrapper.generator.eval()

                cpu_params = 0
                non_cpu_params = 0
                for name, param in generator_wrapper.generator.named_parameters():
                    if param.device.type == "cpu":
                        cpu_params += 1
                    else:
                        non_cpu_params += 1
                        st.warning(
                            f"⚠️ {style_key}: Параметр {name} не на CPU: {param.device}"
                        )
                        param.data = param.data.to(device)

                for name, buffer in generator_wrapper.generator.named_buffers():
                    if buffer.device.type != "cpu":
                        st.warning(
                            f"⚠️ {style_key}: Буфер {name} не на CPU: {buffer.device}"
                        )
                        buffer.data = buffer.data.to(device)

                if non_cpu_params > 0:
                    st.warning(
                        f"⚠️ {style_key}: Найдено {non_cpu_params} параметров не на CPU, исправлено"
                    )

            models[style_key] = {
                "generator": generator_wrapper,
                "name": style_info["name"],
                "model_key": model_key,
                "cos_sim": style_info["cos_sim"],
                "metadata": metadata,
            }


        except Exception as e:
            st.error(f"❌ Не удалось загрузить {model_key}: {e}")
            import traceback

            with st.expander(f"🔍 Детали ошибки загрузки {model_key}"):
                st.code(traceback.format_exc())

    if not models:
        st.error("❌ Не найдено обученных моделей!")
        return None

    return models


def tensor_to_pil(tensor):
    try:
        if tensor.dim() == 4:
            tensor = tensor[0]
        elif tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
        elif tensor.dim() != 3:
            raise ValueError(
                f"Неожиданная размерность тензора: {tensor.dim()}, форма: {tensor.shape}"
            )

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()

        if tensor.dim() != 3:
            raise ValueError(
                f"После обработки тензор должен быть 3D, получено: {tensor.dim()}, форма: {tensor.shape}"
            )

        if tensor.shape[0] > 3:
            tensor = tensor[:3]
        elif tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        elif tensor.shape[0] != 3:
            raise ValueError(
                f"Неожиданное количество каналов: {tensor.shape[0]}, форма: {tensor.shape}"
            )

        if tensor.shape[0] != 3:
            raise ValueError(
                f"Перед permute должно быть 3 канала, получено: {tensor.shape[0]}, форма: {tensor.shape}"
            )

        img_np = (
            ((tensor.permute(1, 2, 0).numpy() + 1) / 2 * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

        if img_np.shape[2] != 3:
            raise ValueError(
                f"После permute должно быть 3 канала, получено: {img_np.shape[2]}, форма: {img_np.shape}"
            )

        return Image.fromarray(img_np)
    except Exception as e:
        error_info = f"Ошибка в tensor_to_pil:\n"
        error_info += f"  Исходная форма тензора: {tensor.shape if hasattr(tensor, 'shape') else 'N/A'}\n"
        error_info += (
            f"  Размерность: {tensor.dim() if hasattr(tensor, 'dim') else 'N/A'}\n"
        )
        error_info += (
            f"  Устройство: {tensor.device if hasattr(tensor, 'device') else 'N/A'}\n"
        )
        error_info += f"  Тип: {type(tensor)}\n"
        error_info += f"  Ошибка: {str(e)}"
        raise ValueError(error_info) from e


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

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🎨 Сгенерировать!", type="primary", width="stretch"):
            with st.spinner("Генерация изображений..."):
                try:
                    generator = selected_model["generator"]
                    device = "cpu"

                    if hasattr(generator, "generator"):
                        generator.generator = generator.generator.to(device)
                        generator.generator.eval()

                    if hasattr(generator, "generator") and hasattr(
                        generator.generator, "synthesis"
                    ):
                        first_param = next(generator.generator.synthesis.parameters())
                        if first_param.device.type != "cpu":
                            st.warning(
                                f"⚠️ Генератор не на CPU! Устройство: {first_param.device}"
                            )
                            generator.generator = generator.generator.to(device)

                    torch.manual_seed(seed)

                    with torch.no_grad():
                        try:
                            z = torch.randn(1, 512, device=device)

                            truncation = 0.8
                            truncation_latent = None
                            if hasattr(generator, "mean_latent"):
                                if isinstance(
                                    generator.mean_latent, torch.Tensor
                                ):
                                    truncation_latent = (
                                        generator.mean_latent.to(device)
                                    )
                                else:
                                    truncation_latent = generator.mean_latent(
                                        n_latent=2048
                                    ).to(device)
                            else:
                                w_avg_samples = []
                                for _ in range(100):
                                    z_sample = torch.randn(
                                        1, 512, device=device
                                    )
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

                            if z.device.type != "cpu":
                                z = z.cpu()
                            if (
                                truncation_latent is not None
                                and truncation_latent.device.type != "cpu"
                            ):
                                truncation_latent = truncation_latent.cpu()

                            if hasattr(generator, "generator"):
                                for param in generator.generator.parameters():
                                    if param.device.type != "cpu":
                                        st.warning(
                                            f"⚠️ Параметр генератора не на CPU: {param.device}"
                                        )
                                        break

                            try:
                                logger.debug(
                                    f"Вызов generator() для стиля {selected_model['name']}"
                                )
                                logger.debug(f"z shape: {z.shape}, z device: {z.device}, z dtype: {z.dtype}")
                                if truncation_latent is not None:
                                    logger.debug(f"truncation_latent shape: {truncation_latent.shape}, device: {truncation_latent.device}, dtype: {truncation_latent.dtype}")
                                
                                try:
                                    image, _ = generator(
                                        [z],
                                        truncation=truncation,
                                        truncation_latent=truncation_latent,
                                    )
                                    logger.debug("Generator вызван успешно")
                                except RuntimeError as rt_error:
                                    error_tb = traceback.format_exc()
                                    logger.error(f"RuntimeError при вызове generator(): {rt_error}\n{error_tb}")
                                    raise
                                except Exception as inner_error:
                                    error_tb = traceback.format_exc()
                                    logger.error(f"Внутренняя ошибка при вызове generator(): {inner_error}\n{error_tb}")
                                    raise
                                
                                if image is None:
                                    raise ValueError("Генератор вернул None")
                                
                                logger.debug(f"Generator вернул результат, тип: {type(image)}")
                                
                                if isinstance(image, (list, tuple)):
                                    if len(image) == 0:
                                        raise ValueError("Генератор вернул пустой список")
                                    img_tensor = image[0]
                                    logger.debug(f"Извлечен img_tensor из списка, форма: {img_tensor.shape if hasattr(img_tensor, 'shape') else 'N/A'}")
                                elif isinstance(image, torch.Tensor):
                                    img_tensor = image
                                    logger.debug(f"Извлечен img_tensor напрямую, форма: {img_tensor.shape}")
                                else:
                                    raise TypeError(f"Неожиданный тип результата генератора: {type(image)}")
                                
                                if not isinstance(img_tensor, torch.Tensor):
                                    raise TypeError(f"Ожидался torch.Tensor, получен {type(img_tensor)}")
                                
                                if img_tensor.device.type != "cpu":
                                    logger.debug(f"Перемещаем img_tensor с {img_tensor.device} на CPU")
                                    img_tensor = img_tensor.cpu()

                                logger.debug(f"Обработка формы тензора: dim={img_tensor.dim()}, shape={img_tensor.shape}")
                                
                                if img_tensor.dim() == 4:
                                    if img_tensor.shape[0] > 1:
                                        logger.debug(f"Батч из {img_tensor.shape[0]} изображений, берем первое")
                                    img_tensor = img_tensor[0]  # [C, H, W]
                                    logger.debug(f"После извлечения из батча: shape={img_tensor.shape}")
                                elif img_tensor.dim() != 3:
                                    error_msg = f"Неожиданная размерность тензора: {img_tensor.dim()}, форма: {img_tensor.shape}"
                                    logger.error(error_msg)
                                    raise ValueError(error_msg)
                                
                                logger.debug(f"Финальная форма перед tensor_to_pil: {img_tensor.shape}")
                                
                                img_pil = tensor_to_pil(img_tensor)
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
                                
                            except Exception as call_error:
                                error_tb = traceback.format_exc()
                                logger.error(
                                    f"Ошибка при вызове generator() для стиля '{selected_model['name']}': {call_error}\n{error_tb}"
                                )
                                st.error(
                                    f"❌ Ошибка при вызове generator() для стиля '{selected_model['name']}': {call_error}"
                                )

                                st.text("Полный traceback:")
                                st.code(error_tb)
                                with st.expander(
                                    f"🔍 Детали ошибки вызова генератора"
                                ):
                                    st.code(error_tb)
                                raise
                        except Exception as gen_error:
                            st.error(
                                f"❌ Ошибка при генерации изображения для стиля '{selected_model['name']}': {gen_error}"
                            )
                            import traceback

                            with st.expander(
                                f"🔍 Детали ошибки генерации"
                            ):
                                st.code(traceback.format_exc())
                            raise

                except Exception as e:
                    error_msg = str(e)
                    error_tb = traceback.format_exc()

                    st.error(
                        f"❌ Ошибка генерации для стиля '{selected_model['name']}': {error_msg}"
                    )

                    st.text("Полный traceback ошибки:")
                    st.code(error_tb)

                    if "cuda" in error_msg.lower() or "device" in error_msg.lower():
                        st.warning(
                            "💡 Похоже на проблему с устройством. Убедитесь, что модель загружена на CPU."
                        )
                    elif "tensor" in error_msg.lower() or "shape" in error_msg.lower():
                        st.warning(
                            "💡 Похоже на проблему с формой тензора или устройством."
                        )
                    elif "dtype" in error_msg.lower() or "float" in error_msg.lower():
                        st.warning(
                            "💡 Похоже на проблему с типом данных (dtype). Попробуйте использовать float32."
                        )

                    with st.expander(
                        f"🔍 Детали ошибки (стиль: {selected_model['name']})"
                    ):
                        st.code(error_tb)

                    st.info(
                        "💡 Попробуйте перезагрузить страницу или выбрать другой стиль."
                    )

    with col2:
        st.info(f"**Стиль:** {selected_model['name']}")
        st.info(f"**Качество:** {selected_model['cos_sim']:.4f}")

        metadata = selected_model.get("metadata", {})
        if metadata:
            target_class = metadata.get("target_class", "N/A")
            st.info(f"**Промпт:** {target_class[:60]}...")

        st.info(f"**Seed:** {seed}")

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
