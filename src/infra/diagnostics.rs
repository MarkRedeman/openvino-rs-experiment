use std::ffi::CStr;

/// Retrieve the last detailed error message from the OpenVINO C runtime.
///
/// The OpenVINO C API stores a thread-local error string whenever an API call
/// fails. The `openvino` Rust crate only surfaces the status-code enum (e.g.
/// `GeneralError`), which is rarely helpful.  This function calls
/// [`ov_get_last_err_msg`] directly to obtain the full C++ exception text.
pub fn get_last_openvino_error() -> String {
    let ptr = unsafe { openvino_sys::ov_get_last_err_msg() };
    if ptr.is_null() {
        return "(no detail available)".to_string();
    }
    let msg = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    if msg.is_empty() {
        "(no detail available)".to_string()
    } else {
        msg
    }
}
