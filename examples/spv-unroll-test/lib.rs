#![no_std]

use spirv_std::spirv;

pub use spirv_std::glam;

// Basic while loop: i < 4
#[spirv(compute(threads(1)))]
pub fn test_unroll(
    #[spirv(global_invocation_id)] _id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32; 64],
) {
    let mut i: u32 = 0;
    while i < 4 {
        data[i as usize] = i;
        i += 1;
    }
}

// for i in 0..4
#[spirv(compute(threads(1)))]
pub fn test_for_range(
    #[spirv(global_invocation_id)] _id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32; 64],
) {
    for i in 0u32..4 {
        data[i as usize] = i * 2;
    }
}

// Accumulator: sum of 0..4
#[spirv(compute(threads(1)))]
pub fn test_accumulate(
    #[spirv(global_invocation_id)] _id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32; 64],
) {
    let mut sum: u32 = 0;
    for i in 0u32..4 {
        sum += i;
    }
    data[0] = sum;
}

// Two independent state variables
#[spirv(compute(threads(1)))]
pub fn test_two_state_vars(
    #[spirv(global_invocation_id)] _id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32; 64],
) {
    let mut i: u32 = 0;
    let mut val: u32 = 10;
    while i < 4 {
        data[i as usize] = val;
        i += 1;
        val += 3;
    }
}

// Nested loops: inner 0..4, outer 0..2
#[spirv(compute(threads(1)))]
pub fn test_nested(
    #[spirv(global_invocation_id)] _id: glam::UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] data: &mut [u32; 64],
) {
    for i in 0u32..2 {
        for j in 0u32..4 {
            data[(i * 4 + j) as usize] = i + j;
        }
    }
}
