//! Camera capture module

use crate::config::VideoConfig;
use crate::error::{Error, Result};
use opencv::{
    core::Mat,
    prelude::*,
    videoio::{self, VideoCapture},
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// Camera wrapper for thread-safe access
pub struct Camera {
    capture: Arc<Mutex<VideoCapture>>,
    #[allow(dead_code)]
    config: VideoConfig,
}

// Safety: VideoCapture is not Send/Sync by default, but we protect it with Mutex
unsafe impl Send for Camera {}
unsafe impl Sync for Camera {}

impl Camera {
    /// Open camera with the given configuration
    pub fn open(config: &VideoConfig) -> Result<Self> {
        let mut capture = open_capture(&config.device)?;

        if !capture.is_opened()? {
            return Err(Error::Camera(format!(
                "Failed to open camera: {}",
                config.device
            )));
        }

        // Set frame dimensions if specified
        if config.frame_width > 0 {
            capture.set(videoio::CAP_PROP_FRAME_WIDTH, config.frame_width as f64)?;
        }
        if config.frame_height > 0 {
            capture.set(videoio::CAP_PROP_FRAME_HEIGHT, config.frame_height as f64)?;
        }

        Ok(Self {
            capture: Arc::new(Mutex::new(capture)),
            config: config.clone(),
        })
    }

    /// Open camera by device index
    pub fn open_by_index(index: i32) -> Result<Self> {
        let capture = VideoCapture::new(index, videoio::CAP_V4L2)?;

        if !capture.is_opened()? {
            return Err(Error::Camera(format!(
                "Failed to open camera at index {}",
                index
            )));
        }

        Ok(Self {
            capture: Arc::new(Mutex::new(capture)),
            config: VideoConfig::default(),
        })
    }

    /// Read a frame from the camera
    pub fn read_frame(&self) -> Result<Mat> {
        let mut capture = self
            .capture
            .lock()
            .map_err(|e| Error::Camera(format!("Failed to lock camera: {}", e)))?;

        let mut frame = Mat::default();
        capture.read(&mut frame)?;

        if frame.empty() {
            return Err(Error::Camera("Captured empty frame".to_string()));
        }

        Ok(frame)
    }

    /// Check if camera is opened
    pub fn is_opened(&self) -> Result<bool> {
        let capture = self
            .capture
            .lock()
            .map_err(|e| Error::Camera(format!("Failed to lock camera: {}", e)))?;
        Ok(capture.is_opened()?)
    }

    /// Release the camera
    pub fn release(&self) -> Result<()> {
        let mut capture = self
            .capture
            .lock()
            .map_err(|e| Error::Camera(format!("Failed to lock camera: {}", e)))?;
        capture.release()?;
        Ok(())
    }

    /// Get current frame width
    pub fn frame_width(&self) -> Result<i32> {
        let capture = self
            .capture
            .lock()
            .map_err(|e| Error::Camera(format!("Failed to lock camera: {}", e)))?;
        Ok(capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32)
    }

    /// Get current frame height
    pub fn frame_height(&self) -> Result<i32> {
        let capture = self
            .capture
            .lock()
            .map_err(|e| Error::Camera(format!("Failed to lock camera: {}", e)))?;
        Ok(capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let _ = self.release();
    }
}

/// List available camera devices
pub fn list_cameras() -> Result<Vec<String>> {
    let mut cameras = Vec::new();

    // Check /dev/video* devices
    for i in 0..10 {
        let device = format!("/dev/video{}", i);
        if std::path::Path::new(&device).exists() {
            // Try to open it to verify it's a valid camera
            if let Ok(cap) = open_capture(&device) {
                if cap.is_opened().unwrap_or(false) {
                    cameras.push(device);
                }
            }
        }
    }

    Ok(cameras)
}

/// Check if a camera device is available
pub fn check_camera(device: &str) -> bool {
    if let Ok(cap) = open_capture(device) {
        cap.is_opened().unwrap_or(false)
    } else {
        false
    }
}

fn open_capture(device: &str) -> Result<VideoCapture> {
    let trimmed = device.trim();

    if trimmed.is_empty() {
        let capture = VideoCapture::new(0, videoio::CAP_V4L2)?;
        if capture.is_opened()? {
            return Ok(capture);
        }
        return Ok(VideoCapture::new(0, videoio::CAP_ANY)?);
    }

    if let Some(index) = parse_camera_index(trimmed) {
        let capture = VideoCapture::new(index, videoio::CAP_V4L2)?;
        if capture.is_opened()? {
            return Ok(capture);
        }
    }

    let capture = VideoCapture::from_file(trimmed, videoio::CAP_ANY)?;
    Ok(capture)
}

fn parse_camera_index(device: &str) -> Option<i32> {
    if let Ok(index) = device.parse::<i32>() {
        return Some(index);
    }

    let candidates = [
        Some(PathBuf::from(device)),
        std::fs::canonicalize(device).ok(),
    ];

    candidates
        .into_iter()
        .flatten()
        .find_map(|path| parse_video_path_index(&path))
}

fn parse_video_path_index(path: &Path) -> Option<i32> {
    let name = path.file_name()?.to_str()?;
    let index = name.strip_prefix("video")?;
    index.parse::<i32>().ok()
}
