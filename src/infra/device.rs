use openvino::DeviceType;

pub fn parse_device(device: &str) -> DeviceType<'static> {
    match device.to_uppercase().as_str() {
        "CPU" => DeviceType::CPU,
        "GPU" => DeviceType::GPU,
        _ => DeviceType::CPU,
    }
}
