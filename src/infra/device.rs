use openvino::DeviceType;
use std::borrow::Cow;

pub fn parse_device(device: &str) -> DeviceType<'static> {
    match device.to_uppercase().as_str() {
        "CPU" => DeviceType::CPU,
        "GPU" => DeviceType::GPU,
        "NPU" => DeviceType::NPU,
        _ => DeviceType::Other(Cow::Owned(device.to_string())),
    }
}
