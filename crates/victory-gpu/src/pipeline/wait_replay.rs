//! Independent replay of #1181 with four controlled pending jobs and one WARP PSO.
use super::*;
use crate::copystat::{C, count};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

fn pending() -> Arc<Pending> {
    let p = Pending::new(Vec::new(), None);
    p.queue_state.store(MAKING, Ordering::Relaxed);
    p.used.store(true, Ordering::Relaxed);
    Arc::new(p)
}

fn next_frame() {
    let previous = crate::frame::current();
    let mut token = 0;
    crate::frame::wait(std::ptr::null(), 0, 0, std::ptr::null(), 2, &mut token);
    assert_eq!(token, previous + 1);
}

fn late_compute_pipeline() -> Option<HostPipeline> {
    let host = crate::host::host()?;
    let shader = crate::host::compile(
        r#"[RootSignature("")] [numthreads(1,1,1)] void main() {}"#,
        s!("main"),
        s!("cs_5_1"),
    )
    .unwrap();
    // SAFETY: the compiled bytecode includes its root signature and all
    // inputs remain live until pipeline creation returns.
    let (root, pso) = unsafe {
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &shader).unwrap();
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: shader.as_ptr().cast(),
                    BytecodeLength: shader.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    };
    Some(HostPipeline {
        compute: true,
        mesh: false,
        pso,
        root,
        stages: Vec::new(),
        slots: 0,
        inputs: Vec::new(),
        rects: false,
        name: "late WARP pipeline".into(),
        state: String::new(),
    })
}

#[test]
fn four_inflight_frames_and_late_completion() {
    const CHILD: &str = "VICTORY_TEST_REPEATED_PIPELINE_WAIT";
    if std::env::var_os(CHILD).is_none() {
        for (shared, budget) in [("1", "20"), ("0", "20"), ("1", "0")] {
            let result = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    std::thread::current().name().unwrap(),
                    "--nocapture",
                ])
                .env(CHILD, shared)
                .env("VICTORY_D3D12_WARP", "1")
                .env("VICTORY_PIPELINE_WAIT_MS", budget)
                .env("VICTORY_PIPELINE_WAIT_SHARED", shared)
                .status()
                .unwrap();
            assert!(result.success());
        }
        return;
    }
    let jobs = std::array::from_fn::<_, 4, _>(|_| pending());
    let late = late_compute_pipeline();
    let waited = || count(C::PipelineWaitUs);
    let budget_us = wait_budget().as_micros() as u64;
    let bind = |p: &Arc<Pending>| {
        let before = waited();
        assert!(ready(p).is_none());
        waited() - before
    };
    let mut frames = Vec::new();
    for _ in 0..6 {
        next_frame();
        let us: Vec<_> = jobs.iter().map(bind).collect();
        frames.push(us);
    }
    println!(
        "shared {}, budget {budget_us} us: wait us per job by frame {frames:?}",
        shared_budget()
    );
    for (frame, jobs) in frames.iter().enumerate() {
        for (job, &micros) in jobs.iter().enumerate() {
            if budget_us > 0 && job == frame {
                assert!(micros >= budget_us);
            } else {
                assert_eq!(micros, 0);
            }
        }
    }
    // A fresh job may take the idle frame's budget. An old recording
    // thread arriving late must not reopen that budget for another job.
    assert!(bind(&pending()) >= budget_us);
    let before = waited();
    wait(&pending(), crate::frame::current() - 1);
    assert_eq!(waited(), before);
    assert_eq!(bind(&pending()), 0);
    if let Some(late) = late {
        let worker = Arc::clone(&jobs[3]);
        std::thread::spawn(move || assert!(worker.made.set(Some(late)).is_ok()))
            .join()
            .unwrap();
        let before = waited();
        assert_eq!(ready(&jobs[3]).unwrap().name, "late WARP pipeline");
        assert_eq!(waited(), before);
        eprintln!("late WARP compute pipeline returned without another wait");
    } else {
        eprintln!("late pipeline check skipped: WARP is unavailable");
    }

    // Concurrent requests for one unfinished job agree on the first frame.
    next_frame();
    let contested = pending();
    let gate = std::sync::Barrier::new(4);
    let before = waited();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                gate.wait();
                assert!(ready(&contested).is_none());
            });
        }
    });
    assert!(waited() - before >= budget_us);
    if budget_us == 0 {
        assert_eq!(waited(), before);
    }
    assert_eq!(
        contested.waited_frame.load(Ordering::Relaxed),
        if budget_us == 0 {
            u64::MAX
        } else {
            crate::frame::current()
        }
    );
    next_frame();
    assert_eq!(bind(&contested), 0);
}
