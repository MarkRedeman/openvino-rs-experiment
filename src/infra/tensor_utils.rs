pub fn cast_bytes_mut_to_f32(bytes: &mut [u8]) -> &mut [f32] {
    assert!(
        bytes.len() % std::mem::size_of::<f32>() == 0,
        "buffer length is not a multiple of f32 size"
    );
    assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<f32>() == 0,
        "buffer is not aligned for f32"
    );
    unsafe {
        std::slice::from_raw_parts_mut(
            bytes.as_mut_ptr() as *mut f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    }
}
