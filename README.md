# FacePass - Linux 人脸识别 PAM 认证系统

FacePass 是一个基于 Rust 的 Linux 人脸识别认证系统，支持 sudo、polkit 等场景的人脸解锁。

## 特性

- 🔐 PAM 集成：支持 sudo、polkit 等需要认证的场景
- 🚀 高性能：使用 Rust 编写，原生 OpenCV 绑定
- 🔒 安全：SSH 会话自动跳过，支持合盖检测
- 🎯 精准：基于 YuNet + SFace + MiniFASNetV2 / MiniFASNetV1SE 活体融合推理
- 🛠️ 易用：CLI 工具管理人脸数据

## 架构

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│ pam_facepass.so │────▶│  facepass-daemon     │◀────│  facepass CLI   │
│  (PAM 模块)     │     │  (守护进程)          │     │  (管理工具)     │
│  轻量级 IPC     │     │  人脸识别 + 摄像头   │     │  add/remove/test│
└─────────────────┘     └──────────────────────┘     └─────────────────┘
        │                        │
        │ Unix Socket            │ 调用
        │ /run/facepass.sock     ▼
        │               ┌──────────────────────┐
        └──────────────▶│  facepass-core       │
                        │  YuNet + SFace       │
                        │  OpenCV 图像处理     │
                        └──────────────────────┘
```

## 安装

### 依赖

- OpenCV 4.x (包含 DNN 模块)
- Rust 工具链

#### Arch Linux
```bash
sudo pacman -S opencv vtk hdf5 fmt
```

#### Debian/Ubuntu
```bash
sudo apt install libopencv-dev
```

### 下载模型

在编译前，需要下载 ONNX 模型文件：

1. [YuNet 人脸检测模型](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
2. [SFace 人脸识别模型](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)
3. `MiniFASNetV2.onnx` 活体检测模型
4. `MiniFASNetV1SE.onnx` 活体检测模型

建议同时准备两个模型并在配置中启用 `fusion`，以匹配训练分布中的 2.7 和 4.0 两种裁切范围。

将模型文件下载到 `models/` 目录：
```bash
mkdir -p models
# 手动下载或使用浏览器保存到 models/ 目录
```

### 编译

```bash
cargo build --release
```

### 安装

```bash
# 使用安装脚本
sudo ./scripts/install.sh

# 或手动安装
sudo install -Dm755 target/release/facepass /usr/local/bin/facepass
sudo install -Dm755 target/release/facepass-daemon /usr/local/bin/facepass-daemon
sudo install -Dm644 target/release/libpam_facepass.so /usr/lib/security/pam_facepass.so
sudo mkdir -p /usr/share/facepass/models
sudo cp models/*.onnx /usr/share/facepass/models/
sudo mkdir -p /etc/facepass
sudo cp config/facepass.toml /etc/facepass/config.toml
```

## 使用

### 1. 启动守护进程

```bash
# 使用 systemd
sudo systemctl start facepass
sudo systemctl enable facepass

# 或手动启动
sudo facepass-daemon
```

### 2. 添加人脸

```bash
sudo facepass add
```

程序会打开摄像头并检测人脸。保持面部正对摄像头，直到成功捕获。

### 3. 测试识别

```bash
sudo facepass test
```

### 4. 配置 PAM

编辑 `/etc/pam.d/sudo`，在开头添加：

```
auth sufficient pam_facepass.so
```

> ⚠️ **警告**: 修改 PAM 配置时请保持一个 root 终端会话，以防配置错误导致无法登录。

### CLI 命令

```bash
# 添加人脸
sudo facepass add [--user <用户名>] [--group <组名或序号>] [--label <标签>] [标签]
sudo facepass add normal

# 列出用户/人脸组/人脸
sudo facepass list [--user <用户名>] [--group <组名或序号>] [--depth <1|2|3>]

# 删除用户、组或人脸
sudo facepass remove [--user <用户名>] [--group <组名或序号>] [--face <标签或序号>]

# 切换默认人脸组
sudo facepass status --set-default-group <组名或序号>
sudo facepass status --set-default-group --user <用户名> --group <组名或序号>
sudo facepass status --set-default-group <组名或序号> --show

# 测试识别
sudo facepass test [--user <用户名>] [--group <组名或序号>] [--frames <帧数>]

# 查看状态
sudo facepass status

# 列出摄像头
sudo facepass cameras
```

## 配置

配置文件位于 `/etc/facepass/config.toml`。

主要配置项：

```toml
[video]
device = "/dev/video0"      # 摄像头设备
timeout = 5                  # 认证超时（秒）

[recognition]
similarity_threshold = 0.4   # 相似度阈值（越高越严格）
max_faces_per_group = 5      # 每个人脸组最大人脸数
consecutive_match_frames = 1 # 需要连续匹配次数

[security]
ignore_ssh = true            # SSH 会话跳过
ignore_closed_lid = true     # 合盖时跳过
```

## API 接口

FacePass 提供多种调用方式，方便其他程序集成。

### 1. Unix Socket IPC

守护进程监听 Unix Socket，其他程序可通过发送 JSON 请求进行认证。

**Socket 路径**: `/run/facepass/facepass.sock`

**请求格式**:
```json
{
  "msg_type": "auth",
  "username": "用户名",
  "source": "调用来源",
  "timeout": 5
}
```

**响应格式**:
```json
{
  "success": true,
  "message": "Face recognized",
  "confidence": 0.85,
  "matched_label": "normal"
}
```

**Python 调用示例**:
```python
import socket
import json

def authenticate(username, timeout=5):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect("/run/facepass/facepass.sock")

    request = {
        "msg_type": "auth",
        "username": username,
        "source": "my_app",
        "timeout": timeout
    }

    sock.send((json.dumps(request) + "\n").encode())
    response = sock.recv(4096).decode()
    sock.close()

    return json.loads(response)

# 使用
result = authenticate("ysltr")
if result["success"]:
    print(f"认证成功! 相似度: {result['confidence']*100:.1f}%")
else:
    print(f"认证失败: {result['message']}")
```

**Bash 调用示例**:
```bash
#!/bin/bash
echo '{"msg_type":"auth","username":"ysltr","source":"script","timeout":5}' | \
  nc -U /run/facepass/facepass.sock
```

**Rust 调用示例**:
```rust
use std::os::unix::net::UnixStream;
use std::io::{Write, BufRead, BufReader};

fn authenticate(username: &str) -> Result<bool, Box<dyn std::error::Error>> {
    let mut stream = UnixStream::connect("/run/facepass/facepass.sock")?;

    let request = format!(
        r#"{{"msg_type":"auth","username":"{}","source":"my_app","timeout":5}}"#,
        username
    );

    stream.write_all(request.as_bytes())?;
    stream.write_all(b"\n")?;

    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response)?;

    // 解析 JSON 响应
    Ok(response.contains(r#""success":true"#))
}
```

### 2. PAM 集成

通过配置 PAM，任何使用 PAM 认证的服务都可以使用人脸识别。

**支持的服务**:
- `sudo` - 管理员命令
- `polkit-1` - 图形化权限提升
- `login` - 终端登录
- `sddm` / `gdm` - 显示管理器
- `su` - 切换用户
- `ssh` (本地) - SSH 本地认证

**配置方法**:

编辑对应服务的 PAM 配置文件，在 `auth` 部分开头添加：

```
auth sufficient pam_facepass.so
```

**配置示例**:

```bash
# sudo
echo "auth sufficient pam_facepass.so" | sudo tee -a /etc/pam.d/sudo

# polkit (图形化密码框)
sudo sed -i '1a auth sufficient pam_facepass.so' /etc/pam.d/polkit-1

# 终端登录
sudo sed -i '1a auth sufficient pam_facepass.so' /etc/pam.d/login
```

### 3. CLI 命令调用

其他程序可以通过执行 CLI 命令来进行认证。

**认证命令**:
```bash
# 测试当前用户的人脸认证（返回退出码表示结果）
facepass test --frames 1

# 检查退出码
if facepass test --frames 1 2>/dev/null; then
    echo "认证成功"
else
    echo "认证失败"
fi
```

**获取状态**:
```bash
# JSON 格式输出状态
facepass status
```

### 4. Systemd 服务

守护进程可作为 systemd 服务运行：

```bash
# 启动服务
sudo systemctl start facepass

# 开机自启
sudo systemctl enable facepass

# 查看日志
journalctl -u facepass -f
```

**服务文件** (`/etc/systemd/system/facepass.service`):
```ini
[Unit]
Description=FacePass Face Authentication Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/facepass-daemon
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

## 开发

### 项目结构

```
facepass/
├── facepass-core/      # 核心库
├── facepass-daemon/    # 守护进程
├── facepass-cli/       # CLI 工具
├── facepass-pam/       # PAM 模块
├── config/             # 配置文件模板
├── models/             # ONNX 模型
└── scripts/            # 安装脚本
```

### 本地测试

使用开发配置：

```bash
# 复制开发配置
cp config/facepass-dev.toml ~/.config/facepass/config.toml

# 运行 CLI
./target/release/facepass --config ~/.config/facepass/config.toml add
```

## 技术栈

- **人脸检测**: YuNet (OpenCV DNN)
- **特征提取**: SFace (128维向量)
- **比对算法**: 余弦相似度
- **IPC**: Unix Domain Socket + JSON
- **数据存储**: bincode 序列化

## 参考项目

- [FaceWinUnlock-Tauri](https://github.com/dc114154qq/FaceWinUnlock-Tauri) - Windows 人脸解锁
- [Howdy](https://github.com/boltgolt/howdy) - Linux 人脸解锁 (Python/dlib)

## 许可证

MIT License
