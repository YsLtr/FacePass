//! FacePass PAM Module
//!
//! This is a lightweight PAM module that communicates with the facepass-daemon
//! via Unix socket to perform face authentication.

mod ipc;

use pam::{export_pam_module, get_user, PamHandle, PamModule, PamReturnCode};
use std::ffi::CStr;
use std::os::raw::c_uint;

/// PAM module struct
pub struct PamFacePass;

impl PamModule for PamFacePass {
    /// Called when PAM wants to authenticate the user
    fn authenticate(handle: &PamHandle, _args: Vec<&CStr>, _flags: c_uint) -> PamReturnCode {
        // Get the username
        let user = match get_user(handle, None) {
            Ok(u) => u.to_string(),
            Err(_) => return PamReturnCode::User_Unknown,
        };

        // Try to authenticate via the daemon
        match ipc::authenticate(&user, "pam", 5) {
            Ok(response) => {
                if response.success {
                    PamReturnCode::Success
                } else {
                    // Return Auth_Err to indicate face auth failed
                    PamReturnCode::Auth_Err
                }
            }
            Err(_) => {
                // Daemon not available or error - fall through to next module
                PamReturnCode::Ignore
            }
        }
    }

    /// Called to set credentials (we don't need this)
    fn set_credentials(_handle: &PamHandle, _args: Vec<&CStr>, _flags: c_uint) -> PamReturnCode {
        PamReturnCode::Success
    }

    /// Called for account management (we don't need this)
    fn account_management(_handle: &PamHandle, _args: Vec<&CStr>, _flags: c_uint) -> PamReturnCode {
        PamReturnCode::Success
    }
}

export_pam_module!(PamFacePass);
