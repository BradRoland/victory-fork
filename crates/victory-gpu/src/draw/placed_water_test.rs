//! Opt-in D3D12 check of placed UAV writes landing in CPU-visible water memory.
//! Run with `cargo test -p victory-gpu --locked --features diagnostics
//! placed_water_feedback -- --ignored --nocapture`. Set `VICTORY_D3D12_WARP=1`
//! to select WARP, or leave it unset for the default hardware device.

use super::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_FLOAT;
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
};

#[test]
#[ignore = "opt-in placed UAV feedback check on the selected adapter"]
fn placed_water_feedback() {
    // The readback kinds and D3D12 device are process-wide. A fresh process
    // also limits the lifetime of the cached placed resource to this test.
    const CHILD: &str = "VICTORY_TEST_PLACED_WATER_FEEDBACK";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                std::thread::current().name().unwrap(),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_TEXTURE_BACK", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    let host = crate::host::host().expect("a D3D12 device");
    // SAFETY: obtain the adapter that actually created this live device.
    let adapter = unsafe {
        CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .unwrap()
            .EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid())
            .unwrap()
            .GetDesc1()
            .unwrap()
    };
    println!(
        "adapter {}",
        String::from_utf16_lossy(&adapter.Description).trim_end_matches('\0')
    );

    const SIDE: u32 = 256;
    let format = DXGI_FORMAT_R32_FLOAT.0 as u32;
    // Title resource description: one mip and one slice; no UAV resource flag.
    let resource = [3, 0, 0, 0, SIDE, 0, SIDE, 0x1_0001, format, 1, 0, 0, 0, 0];
    let mut bytes = vec![0x4au8; crate::resource::placed_bytes(&resource).unwrap() as usize];
    let at = bytes.as_mut_ptr() as usize;
    let mut words = [0u32; 16];
    words[0] = format;
    words[1] = D3D12_UAV_DIMENSION_TEXTURE2D.0 as u32;
    words[10] = at as u32;
    words[11] = (at >> 32) as u32;

    let mut slot = [0u32; 8];
    let handle = slot.as_mut_ptr() as u64;
    crate::descriptor::create_placed_uav(std::ptr::null(), &resource, &words, handle);
    let gpu = crate::descriptor::host::HostHeap::new(false, 1).unwrap();
    let placed = gpu.mirror(0, handle, true).unwrap();
    let list = host.new_list().unwrap();
    readable(&list, ResourceType::Uav, &placed);
    let Some(View::Unordered(id, _, _)) = crate::descriptor::host::placed_texture(&placed) else {
        panic!("the placed R32_FLOAT texture must resolve to a host UAV");
    };
    let copy = crate::resource::host_copy(id as This).unwrap();
    assert_eq!(gpu.shows(0, &copy.texture), Some(true));
    let (layout, address, size) = crate::resource::title_texture(id as This).unwrap();
    assert_eq!((address, size), (at, bytes.len()));

    // Clear through the mirrored GPU descriptor. D3D12 also requires an
    // identical descriptor in a non-shader-visible heap.
    // SAFETY: the descriptor heap has one slot of the requested type.
    let cpu: ID3D12DescriptorHeap = unsafe {
        host.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap()
    };
    // SAFETY: the heaps, resource, descriptors and CPU allocation remain
    // live through execution. The resource and view have the same format.
    unsafe {
        let native = words
            .as_ptr()
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned();
        let handle = cpu.GetCPUDescriptorHandleForHeapStart();
        host.device.CreateUnorderedAccessView(
            &copy.texture,
            None::<&ID3D12Resource>,
            Some(&native),
            handle,
        );
        list.list.SetDescriptorHeaps(&[Some(gpu.heap.clone())]);
        list.list.ClearUnorderedAccessViewFloat(
            gpu.gpu_handle(0),
            handle,
            &copy.texture,
            &[4.; 4],
            &[],
        );
    }
    list.close();
    host.execute(&[&list]);

    for y in 0..SIDE {
        for x in 0..SIDE {
            let offset = layout.offset(0, x, y, 0).unwrap() as usize;
            assert_eq!(
                bytes[offset..offset + 4],
                4f32.to_le_bytes(),
                "texel ({x}, {y})"
            );
        }
    }
    println!("{} R32_FLOAT texels landed in title memory", SIDE * SIDE);
}
