#!/usr/bin/env python3
"""
Скрипт для применения CPU-поддержки к файлам stylegan-nada после клонирования.
Проверяет устройство и применяет изменения только если используется CPU.
"""

import os
import sys
import torch
import shutil


def check_cpu_device():
    return not torch.cuda.is_available()


def apply_cpu_support_to_fused_act(file_path):
    print(f"Применение CPU-поддержки к {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Проверяем, не применены ли уже изменения
    if "Only load CUDA extensions if CUDA is available" in content:
        print(f"CPU-поддержка уже применена к {file_path}")
        return True

    # Создаем резервную копию
    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"Создана резервная копия: {backup_path}")

    # Заменяем импорт и загрузку расширений
    old_import = """from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)"""

    new_import = """# Only load CUDA extensions if CUDA is available
fused = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        fused = load(
            "fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
    except Exception as e:
        print(f"Warning: Could not load CUDA extension for fused_act: {e}")
        print("Falling back to CPU implementation")
        fused = None
else:
    # CUDA not available, will use CPU fallback
    fused = None"""

    content = content.replace(old_import, new_import)

    # Обновляем FusedLeakyReLUFunctionBackward.forward
    old_forward = """class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias"""

    new_forward = """class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        if fused is None:
            # CPU fallback
            if bias:
                rest_dim = [1] * (grad_output.ndim - 1)
                grad_input = F.leaky_relu(
                    grad_output + out.view(1, out.shape[0], *rest_dim), 
                    negative_slope=negative_slope
                ) * scale
                dim = [0]
                if grad_input.ndim > 2:
                    dim += list(range(2, grad_input.ndim))
                grad_bias = grad_input.sum(dim).detach()
            else:
                grad_input = F.leaky_relu(grad_output, negative_slope=negative_slope) * scale
                grad_bias = grad_output.new_empty(0)
            return grad_input, grad_bias

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias"""

    content = content.replace(old_forward, new_forward)

    # Обновляем backward
    old_backward = """    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None, None"""

    new_backward = """    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        
        if fused is None:
            # CPU fallback - simplified backward
            if gradgrad_bias is not None:
                rest_dim = [1] * (gradgrad_input.ndim - 1)
                gradgrad_out = F.leaky_relu(
                    gradgrad_input + gradgrad_bias.view(1, gradgrad_bias.shape[0], *rest_dim),
                    negative_slope=ctx.negative_slope
                ) * ctx.scale
            else:
                gradgrad_out = F.leaky_relu(gradgrad_input, negative_slope=ctx.negative_slope) * ctx.scale
            return gradgrad_out, None, None, None, None
        
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None, None"""

    content = content.replace(old_backward, new_backward)

    # Обновляем FusedLeakyReLUFunction.forward
    old_func_forward = """class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)

        ctx.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out"""

    new_func_forward = """class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        ctx.bias = bias is not None
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        if fused is None:
            # CPU fallback
            if bias is not None:
                rest_dim = [1] * (input.ndim - bias.ndim - 1)
                out = F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), 
                    negative_slope=negative_slope
                ) * scale
            else:
                out = F.leaky_relu(input, negative_slope=negative_slope) * scale
            ctx.save_for_backward(out)
            return out

        empty = input.new_empty(0)

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)

        return out"""

    content = content.replace(old_func_forward, new_func_forward)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"CPU-поддержка применена к {file_path}")
    return True


def apply_cpu_support_to_upfirdn2d(file_path):
    print(f"Применение CPU-поддержки к {file_path}...")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if "Only load CUDA extensions if CUDA is available" in content:
        print(f"CPU-поддержка уже применена к {file_path}")
        return True

    backup_path = file_path + ".backup"
    shutil.copy2(file_path, backup_path)
    print(f"  Создана резервная копия: {backup_path}")

    old_import = """from torch.utils.cpp_extension import load


module_path = os.path.dirname(__file__)
upfirdn2d_op = load(
    "upfirdn2d",
    sources=[
        os.path.join(module_path, "upfirdn2d.cpp"),
        os.path.join(module_path, "upfirdn2d_kernel.cu"),
    ],
)"""

    new_import = """# Only load CUDA extensions if CUDA is available
upfirdn2d_op = None
if torch.cuda.is_available():
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        upfirdn2d_op = load(
            "upfirdn2d",
            sources=[
                os.path.join(module_path, "upfirdn2d.cpp"),
                os.path.join(module_path, "upfirdn2d_kernel.cu"),
            ],
        )
    except Exception as e:
        print(f"Warning: Could not load CUDA extension for upfirdn2d: {e}")
        print("Falling back to CPU implementation")
        upfirdn2d_op = None
else:
    # CUDA not available, will use CPU fallback
    upfirdn2d_op = None"""

    content = content.replace(old_import, new_import)

    old_backward_forward_start = """class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        ctx.save_for_backward(kernel)

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        return grad_input"""

    new_backward_forward_start = """class UpFirDn2dBackward(Function):
    @staticmethod
    def forward(
        ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
    ):

        up_x, up_y = up
        down_x, down_y = down
        g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad

        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        ctx.save_for_backward(kernel)
        ctx.up_x = up_x
        ctx.up_y = up_y
        ctx.down_x = down_x
        ctx.down_y = down_y
        ctx.pad_x0 = pad_x0
        ctx.pad_x1 = pad_x1
        ctx.pad_y0 = pad_y0
        ctx.pad_y1 = pad_y1
        ctx.in_size = in_size
        ctx.out_size = out_size

        if upfirdn2d_op is None:
            # CPU fallback - use native implementation
            # Reshape grad_output to match expected input format
            grad_output_reshaped = grad_output.reshape(in_size[0], in_size[1], out_size[0], out_size[1])
            grad_input = upfirdn2d_native(
                grad_output_reshaped,
                grad_kernel,
                down_x, down_y, up_x, up_y,
                g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1
            )
            return grad_input

        grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)

        grad_input = upfirdn2d_op.upfirdn2d(
            grad_output,
            grad_kernel,
            down_x,
            down_y,
            up_x,
            up_y,
            g_pad_x0,
            g_pad_x1,
            g_pad_y0,
            g_pad_y1,
        )
        grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])

        return grad_input"""

    content = content.replace(old_backward_forward_start, new_backward_forward_start)

    old_backward_backward = """    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None"""

    new_backward_backward = """    @staticmethod
    def backward(ctx, gradgrad_input):
        kernel, = ctx.saved_tensors

        if upfirdn2d_op is None:
            # CPU fallback
            gradgrad_out = upfirdn2d_native(
                gradgrad_input,
                kernel,
                ctx.up_x, ctx.up_y, ctx.down_x, ctx.down_y,
                ctx.pad_x0, ctx.pad_x1, ctx.pad_y0, ctx.pad_y1
            )
            gradgrad_out = gradgrad_out.view(
                ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
            )
            return gradgrad_out, None, None, None, None, None, None, None, None

        gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)

        gradgrad_out = upfirdn2d_op.upfirdn2d(
            gradgrad_input,
            kernel,
            ctx.up_x,
            ctx.up_y,
            ctx.down_x,
            ctx.down_y,
            ctx.pad_x0,
            ctx.pad_x1,
            ctx.pad_y0,
            ctx.pad_y1,
        )
        # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
        gradgrad_out = gradgrad_out.view(
            ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
        )

        return gradgrad_out, None, None, None, None, None, None, None, None"""

    content = content.replace(old_backward_backward, new_backward_backward)

    old_upfirdn2d_forward = """class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        input = input.reshape(-1, in_h, in_w, 1)

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out"""

    new_upfirdn2d_forward = """class UpFirDn2d(Function):
    @staticmethod
    def forward(ctx, input, kernel, up, down, pad):
        up_x, up_y = up
        down_x, down_y = down
        pad_x0, pad_x1, pad_y0, pad_y1 = pad

        kernel_h, kernel_w = kernel.shape
        batch, channel, in_h, in_w = input.shape
        ctx.in_size = input.shape

        ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))

        out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
        out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
        ctx.out_size = (out_h, out_w)

        ctx.up = (up_x, up_y)
        ctx.down = (down_x, down_y)
        ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)

        g_pad_x0 = kernel_w - pad_x0 - 1
        g_pad_y0 = kernel_h - pad_y0 - 1
        g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
        g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1

        ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)

        if upfirdn2d_op is None:
            # CPU fallback - use native implementation
            out = upfirdn2d_native(
                input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
            )
            return out

        input = input.reshape(-1, in_h, in_w, 1)

        out = upfirdn2d_op.upfirdn2d(
            input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
        )
        # out = out.view(major, out_h, out_w, minor)
        out = out.view(-1, channel, out_h, out_w)

        return out"""

    content = content.replace(old_upfirdn2d_forward, new_upfirdn2d_forward)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"CPU-поддержка применена к {file_path}")
    return True


def setup_cpu_support(stylegan_nada_dir=None):
    if not check_cpu_device():
        print("Устройство: GPU (CUDA доступна)")
        print("CPU-поддержка не требуется. Пропускаем применение изменений.")
        return True

    print("Устройство: CPU")
    print("Применение CPU-поддержки к файлам stylegan-nada...")

    if stylegan_nada_dir is None:
        possible_paths = [
            "../stylegan_nada",
            "stylegan_nada",
            os.path.join(os.path.dirname(__file__), "stylegan_nada"),
        ]

        for path in possible_paths:
            abs_path = os.path.abspath(path)
            if os.path.exists(abs_path) and os.path.exists(
                os.path.join(abs_path, "ZSSGAN")
            ):
                stylegan_nada_dir = abs_path
                break

        if stylegan_nada_dir is None:
            print("Ошибка: не найдена директория stylegan_nada!")
            print("Укажите путь явно: setup_cpu_support('/path/to/stylegan_nada')")
            return False

    stylegan_nada_dir = os.path.abspath(stylegan_nada_dir)

    if not os.path.exists(stylegan_nada_dir):
        print(f"Ошибка: директория не существует: {stylegan_nada_dir}")
        return False

    if not os.path.exists(os.path.join(stylegan_nada_dir, "ZSSGAN")):
        print(f"Ошибка: ZSSGAN не найден в {stylegan_nada_dir}")
        return False

    print(f"Найдена директория stylegan_nada: {stylegan_nada_dir}")

    fused_act_path = os.path.join(stylegan_nada_dir, "ZSSGAN", "op", "fused_act.py")
    upfirdn2d_path = os.path.join(stylegan_nada_dir, "ZSSGAN", "op", "upfirdn2d.py")

    if not os.path.exists(fused_act_path):
        print(f"Ошибка: файл не найден: {fused_act_path}")
        return False

    if not os.path.exists(upfirdn2d_path):
        print(f"Ошибка: файл не найден: {upfirdn2d_path}")
        return False

    try:
        apply_cpu_support_to_fused_act(fused_act_path)
        apply_cpu_support_to_upfirdn2d(upfirdn2d_path)
        print("CPU-поддержка успешно применена ко всем файлам!")
        return True
    except Exception as e:
        print(f"Ошибка при применении CPU-поддержки: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Применяет CPU-поддержку к stylegan-nada"
    )
    parser.add_argument(
        "--stylegan-nada-dir",
        type=str,
        default=None,
        help="Путь к директории stylegan_nada (по умолчанию ищет автоматически)",
    )

    args = parser.parse_args()

    success = setup_cpu_support(args.stylegan_nada_dir)
    sys.exit(0 if success else 1)
