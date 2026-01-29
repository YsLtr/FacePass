#!/bin/bash
# FacePass 模型下载脚本
# 从 OpenCV Zoo 下载 YuNet 和 SFace ONNX 模型

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

# 创建模型目录
mkdir -p "$MODELS_DIR"

echo "=== FacePass 模型下载脚本 ==="
echo "模型将保存到: $MODELS_DIR"
echo ""

# YuNet 人脸检测模型
YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_FILE="$MODELS_DIR/face_detection_yunet_2023mar.onnx"

# SFace 人脸识别模型
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
SFACE_FILE="$MODELS_DIR/face_recognition_sface_2021dec.onnx"

# 备用下载链接 (Hugging Face)
YUNET_URL_ALT="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
SFACE_URL_ALT="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"

download_file() {
    local url="$1"
    local output="$2"
    local name="$3"
    local alt_url="$4"

    if [ -f "$output" ]; then
        local size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output")
        if [ "$size" -gt 10000 ]; then
            echo "✓ $name 已存在 ($size bytes)"
            return 0
        fi
        echo "! $name 文件损坏，重新下载..."
        rm -f "$output"
    fi

    echo "下载 $name ..."

    # 尝试主要链接
    if command -v wget &> /dev/null; then
        if wget -q --show-progress -O "$output" "$url" 2>/dev/null; then
            echo "✓ $name 下载完成"
            return 0
        fi
        # 尝试备用链接
        if [ -n "$alt_url" ] && wget -q --show-progress -O "$output" "$alt_url" 2>/dev/null; then
            echo "✓ $name 从备用链接下载完成"
            return 0
        fi
    fi

    if command -v curl &> /dev/null; then
        if curl -L -# -o "$output" "$url" 2>/dev/null; then
            echo "✓ $name 下载完成"
            return 0
        fi
        # 尝试备用链接
        if [ -n "$alt_url" ] && curl -L -# -o "$output" "$alt_url" 2>/dev/null; then
            echo "✓ $name 从备用链接下载完成"
            return 0
        fi
    fi

    echo "✗ $name 下载失败"
    echo "  请手动下载:"
    echo "  主链接: $url"
    echo "  备用链接: $alt_url"
    echo "  保存到: $output"
    return 1
}

# 下载模型
download_file "$YUNET_URL" "$YUNET_FILE" "YuNet 人脸检测模型" "$YUNET_URL_ALT"
download_file "$SFACE_URL" "$SFACE_FILE" "SFace 人脸识别模型" "$SFACE_URL_ALT"

echo ""
echo "=== 模型文件状态 ==="
ls -lh "$MODELS_DIR"/*.onnx 2>/dev/null || echo "没有找到模型文件"

echo ""
echo "如果下载失败，请手动下载模型文件:"
echo "1. YuNet: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
echo "2. SFace: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
echo ""
echo "将文件保存到: $MODELS_DIR/"
