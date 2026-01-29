#!/bin/bash
# FacePass 安装脚本
# 安装 FacePass 人脸识别 PAM 认证系统

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_PREFIX="${INSTALL_PREFIX:-/usr/local}"

echo -e "${GREEN}=== FacePass 安装脚本 ===${NC}"
echo "项目目录: $PROJECT_DIR"
echo "安装前缀: $INSTALL_PREFIX"
echo ""

# 检查是否有 root 权限
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${YELLOW}警告: 某些安装步骤需要 root 权限${NC}"
        echo "建议使用: sudo $0"
        echo ""
    fi
}

# 安装二进制文件
install_binaries() {
    echo "安装二进制文件..."

    local release_dir="$PROJECT_DIR/target/release"

    if [ ! -f "$release_dir/facepass" ]; then
        echo -e "${RED}错误: 找不到编译后的二进制文件${NC}"
        echo "请先运行: cargo build --release"
        exit 1
    fi

    sudo install -Dm755 "$release_dir/facepass" "$INSTALL_PREFIX/bin/facepass"
    sudo install -Dm755 "$release_dir/facepass-daemon" "$INSTALL_PREFIX/bin/facepass-daemon"

    echo -e "${GREEN}✓ 二进制文件已安装到 $INSTALL_PREFIX/bin/${NC}"
}

# 安装 PAM 模块
install_pam() {
    echo "安装 PAM 模块..."

    local pam_so="$PROJECT_DIR/target/release/libpam_facepass.so"
    local pam_dir="/usr/lib/security"

    # 某些系统使用不同的路径
    if [ ! -d "$pam_dir" ]; then
        pam_dir="/lib/security"
    fi
    if [ ! -d "$pam_dir" ]; then
        pam_dir="/lib64/security"
    fi

    if [ ! -f "$pam_so" ]; then
        echo -e "${RED}错误: 找不到 PAM 模块${NC}"
        exit 1
    fi

    sudo install -Dm644 "$pam_so" "$pam_dir/pam_facepass.so"

    echo -e "${GREEN}✓ PAM 模块已安装到 $pam_dir/${NC}"
}

# 安装模型文件
install_models() {
    echo "安装模型文件..."

    local models_src="$PROJECT_DIR/models"
    local models_dst="/usr/share/facepass/models"

    if [ ! -f "$models_src/face_detection_yunet_2023mar.onnx" ] || \
       [ ! -f "$models_src/face_recognition_sface_2021dec.onnx" ]; then
        echo -e "${YELLOW}警告: 模型文件不存在${NC}"
        echo "请下载模型文件到 $models_src/"
        echo ""
        echo "下载链接:"
        echo "1. YuNet: https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        echo "2. SFace: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        echo ""
        return 1
    fi

    sudo mkdir -p "$models_dst"
    sudo install -Dm644 "$models_src/face_detection_yunet_2023mar.onnx" "$models_dst/"
    sudo install -Dm644 "$models_src/face_recognition_sface_2021dec.onnx" "$models_dst/"

    echo -e "${GREEN}✓ 模型文件已安装到 $models_dst/${NC}"
}

# 安装配置文件
install_config() {
    echo "安装配置文件..."

    local config_src="$PROJECT_DIR/config/facepass.toml"
    local config_dst="/etc/facepass/config.toml"

    if [ -f "$config_dst" ]; then
        echo -e "${YELLOW}配置文件已存在，跳过${NC}"
        return 0
    fi

    sudo mkdir -p /etc/facepass
    sudo install -Dm644 "$config_src" "$config_dst"

    echo -e "${GREEN}✓ 配置文件已安装到 $config_dst${NC}"
}

# 安装 systemd 服务
install_service() {
    echo "安装 systemd 服务..."

    local service_src="$PROJECT_DIR/config/facepass.service"
    local service_dst="/etc/systemd/system/facepass.service"

    if [ ! -f "$service_src" ]; then
        echo -e "${YELLOW}警告: systemd 服务文件不存在${NC}"
        return 1
    fi

    sudo install -Dm644 "$service_src" "$service_dst"
    sudo systemctl daemon-reload

    echo -e "${GREEN}✓ systemd 服务已安装${NC}"
    echo "  启动服务: sudo systemctl start facepass"
    echo "  开机自启: sudo systemctl enable facepass"
}

# 创建必要目录
create_directories() {
    echo "创建数据目录..."

    sudo mkdir -p /var/lib/facepass/faces
    sudo mkdir -p /run/facepass

    # 设置权限
    sudo chmod 755 /var/lib/facepass
    sudo chmod 700 /var/lib/facepass/faces
    sudo chmod 755 /run/facepass

    echo -e "${GREEN}✓ 数据目录已创建${NC}"
}

# 配置 PAM
setup_pam() {
    echo ""
    echo -e "${YELLOW}PAM 配置说明:${NC}"
    echo ""
    echo "要启用 FacePass 人脸识别，需要修改 PAM 配置文件。"
    echo ""
    echo "对于 sudo，编辑 /etc/pam.d/sudo，在开头添加:"
    echo "  auth sufficient pam_facepass.so"
    echo ""
    echo "对于 polkit，编辑 /etc/pam.d/polkit-1，在开头添加:"
    echo "  auth sufficient pam_facepass.so"
    echo ""
    echo -e "${RED}警告: 配置 PAM 时请小心，错误的配置可能导致无法登录!${NC}"
    echo "建议保持一个 root 终端会话作为备份。"
}

# 显示使用说明
show_usage() {
    echo ""
    echo -e "${GREEN}=== 安装完成 ===${NC}"
    echo ""
    echo "使用方法:"
    echo "  1. 启动守护进程:"
    echo "     sudo systemctl start facepass"
    echo "     或: sudo facepass-daemon"
    echo ""
    echo "  2. 添加人脸:"
    echo "     sudo facepass add"
    echo ""
    echo "  3. 测试识别:"
    echo "     sudo facepass test"
    echo ""
    echo "  4. 列出已添加的人脸:"
    echo "     sudo facepass list"
    echo ""
    echo "  5. 配置 PAM (见上方说明)"
    echo ""
}

# 主函数
main() {
    check_root

    case "${1:-install}" in
        install)
            install_binaries
            install_pam
            install_models || true
            install_config
            install_service || true
            create_directories
            setup_pam
            show_usage
            ;;
        binaries)
            install_binaries
            ;;
        pam)
            install_pam
            ;;
        models)
            install_models
            ;;
        config)
            install_config
            ;;
        service)
            install_service
            ;;
        help|--help|-h)
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  install   完整安装 (默认)"
            echo "  binaries  只安装二进制文件"
            echo "  pam       只安装 PAM 模块"
            echo "  models    只安装模型文件"
            echo "  config    只安装配置文件"
            echo "  service   只安装 systemd 服务"
            echo "  help      显示帮助"
            ;;
        *)
            echo -e "${RED}未知命令: $1${NC}"
            echo "使用 '$0 help' 查看帮助"
            exit 1
            ;;
    esac
}

main "$@"
