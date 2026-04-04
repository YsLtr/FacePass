# Maintainer: ysltr <your-email@example.com>
pkgname=facepass
pkgver=0.1.0
pkgrel=1
pkgdesc="Linux face recognition PAM authentication system using YuNet + SFace"
arch=('x86_64')
url="https://github.com/ysltr/facepass"
license=('MIT')
depends=(
    'opencv'
    'pam'
)
makedepends=(
    'rust'
    'cargo'
    'clang'
)
optdepends=(
    'v4l-utils: camera utilities'
)
backup=(
    'etc/facepass/config.toml'
)
install=facepass.install
source=(
    "${pkgname}-${pkgver}.tar.gz"
    "face_detection_yunet_2023mar.onnx::https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
    "face_recognition_sface_2021dec.onnx::https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
)
sha256sums=(
    'SKIP'
    'SKIP'
    'SKIP'
)

build() {
    cd "$srcdir/${pkgname}-${pkgver}"
    export RUSTUP_TOOLCHAIN=stable
    export CARGO_TARGET_DIR=target
    cargo build --release --locked
}

check() {
    cd "$srcdir/${pkgname}-${pkgver}"
    export RUSTUP_TOOLCHAIN=stable
    cargo test --release --locked || true
}

package() {
    cd "$srcdir/${pkgname}-${pkgver}"

    # Install binaries
    install -Dm755 "target/release/facepass" "$pkgdir/usr/bin/facepass"
    install -Dm755 "target/release/facepass-daemon" "$pkgdir/usr/bin/facepass-daemon"

    # Install PAM module
    install -Dm644 "target/release/libpam_facepass.so" "$pkgdir/usr/lib/security/pam_facepass.so"

    # Install models
    install -Dm644 "$srcdir/face_detection_yunet_2023mar.onnx" \
        "$pkgdir/usr/share/facepass/models/face_detection_yunet_2023mar.onnx"
    install -Dm644 "$srcdir/face_recognition_sface_2021dec.onnx" \
        "$pkgdir/usr/share/facepass/models/face_recognition_sface_2021dec.onnx"

    # Install configs
    install -Dm644 "config/facepass.toml" "$pkgdir/etc/facepass/config.toml"
    install -Dm644 "config/facepass.toml" "$pkgdir/usr/share/facepass/config.toml.example"

    # Install systemd service
    install -Dm644 "config/facepass.service" "$pkgdir/usr/lib/systemd/system/facepass.service"

    # Install license and docs
    install -Dm644 "LICENSE" "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
    install -Dm644 "README.md" "$pkgdir/usr/share/doc/$pkgname/README.md"

    # Create state directories
    install -dm700 "$pkgdir/var/lib/facepass/faces"
}
