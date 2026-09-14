// The trace files are the cross-language oracle. The reference implementation
// writes the cases, this test replays them on every backend, and the comparison
// is on raw bits. Generate the files with:
//
//   go run -tags=fixed_satcounter github.com/dhannyell/fixed/cmd/fixedtrace@<tag> -o build/traces

mod common;

use common::{Gpu, pipeline, read_u32s, readback_buffer, shader, storage_buffer};
use std::path::PathBuf;

const CORE: &str = include_str!("../fixed_core.wgsl");
const PORTABLE: &str = include_str!("../fixed_portable.wgsl");
const INT64: &str = include_str!("../fixed_int64.wgsl");
const BATCH16: &str = include_str!("../batch16.wgsl");
const BATCH48: &str = include_str!("../batch48.wgsl");

const SAT_BINDING: u32 = 3;

const OPS16: [&str; 11] = [
    "add16",
    "sub16",
    "mul16",
    "mul_round16",
    "div16",
    "sqrt16",
    "min16",
    "max16",
    "greater16",
    "equals16",
    "blend16",
];

const OPS48: [&str; 8] = [
    "add48",
    "sub48",
    "mul_add48",
    "to16_48",
    "to16_32",
    "to48_16",
    "to48_32",
    "to32_48",
];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Q16,
    Q32,
    Q48,
    Mask,
    Sat,
}

impl Kind {
    fn parse(name: &str) -> Kind {
        match name {
            "q16" => Kind::Q16,
            "q32" => Kind::Q32,
            "q48" => Kind::Q48,
            "mask" => Kind::Mask,
            "sat" => Kind::Sat,
            other => panic!("unknown column type {other:?}"),
        }
    }

    // How many 32-bit words the value occupies in a storage buffer.
    fn words(self) -> usize {
        match self {
            Kind::Q16 | Kind::Mask => 1,
            Kind::Q32 | Kind::Q48 => 2,
            Kind::Sat => 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Module {
    Batch16,
    Batch48,
}

struct Trace {
    inputs: Vec<Kind>,
    outputs: Vec<Kind>,
    rows: Vec<Vec<u64>>,
}

#[test]
fn batch16_traces_match_the_reference() {
    for op in OPS16 {
        check(op, Module::Batch16);
    }
}

#[test]
fn batch48_traces_match_the_reference() {
    for op in OPS48 {
        check(op, Module::Batch48);
    }
}

fn check(entry_point: &str, module: Module) {
    let trace = read_trace(entry_point);

    for native in [false, true] {
        let source = source(module, native);
        common::for_each_gpu(native, |gpu| {
            check_on(gpu, &source, native, entry_point, module, &trace)
        });
    }
}

fn check_on(
    gpu: &Gpu,
    source: &str,
    native: bool,
    entry_point: &str,
    module: Module,
    trace: &Trace,
) {
    let backend = gpu.backend;
    let where_ = format!("{entry_point} on {backend} native={native}");

    let (got, sat) = dispatch(gpu, source, entry_point, module, trace, &trace.rows);
    let want = expected_words(trace, &trace.rows);

    // Report the first divergence; whole vectors flood the output.
    let mismatch = got.iter().zip(&want).position(|(g, w)| g != w);
    assert!(
        mismatch.is_none() && got.len() == want.len(),
        "{where_}: raw bits diverged at word {:?} of case {:?} (got {:#x?}, want {:#x?})",
        mismatch,
        mismatch.map(|i| i / trace.output_words()),
        mismatch.map(|i| got[i]),
        mismatch.map(|i| want[i]),
    );

    assert_eq!(sat[1], 0, "{where_}: the shader reported a fault");

    let want_sat = expected_saturations(trace, &trace.rows);
    if sat[0] != want_sat {
        // The kernel reports a total per dispatch, never per case. Halving the
        // trace turns that total into the case that produced it.
        let case = locate_saturation(gpu, source, entry_point, module, trace, &trace.rows);
        panic!(
            "{where_}: saturation count diverged (got {}, want {want_sat}); \
             first case with a wrong count is {}",
            sat[0],
            trace.format_case(&case),
        );
    }
}

/// Narrows a saturation divergence down to one case by halving the trace.
fn locate_saturation(
    gpu: &Gpu,
    source: &str,
    entry_point: &str,
    module: Module,
    trace: &Trace,
    rows: &[Vec<u64>],
) -> Vec<u64> {
    if rows.len() == 1 {
        return rows[0].clone();
    }

    let half = rows.len() / 2;
    for part in [&rows[..half], &rows[half..]] {
        let (_, sat) = dispatch(gpu, source, entry_point, module, trace, part);
        if sat[0] != expected_saturations(trace, part) {
            return locate_saturation(gpu, source, entry_point, module, trace, part);
        }
    }

    // Both halves agree while the whole does not. The counter is not additive.
    rows[0].clone()
}

fn dispatch(
    gpu: &Gpu,
    source: &str,
    entry_point: &str,
    module: Module,
    trace: &Trace,
    rows: &[Vec<u64>],
) -> (Vec<u32>, [u32; 2]) {
    let device = &gpu.device;
    let queue = &gpu.queue;

    let module_handle = shader(device, source);
    let pipeline = pipeline(device, &module_handle, entry_point);

    let out_bytes = (rows.len() * trace.output_words() * 4) as u64;
    let out = storage_buffer(device, "out", out_bytes);
    let sat_out = storage_buffer(device, "sat_out", 8);
    queue.write_buffer(&sat_out, 0, &[0; 8]);

    let mut seen: Vec<Kind> = Vec::new();
    let inputs: Vec<(u32, wgpu::Buffer)> = trace
        .inputs
        .iter()
        .enumerate()
        .map(|(column, &kind)| {
            let nth = seen.iter().filter(|&&k| k == kind).count();
            seen.push(kind);

            let bytes: Vec<u8> = rows
                .iter()
                .flat_map(|row| value_words(kind, row[column]))
                .flat_map(u32::to_le_bytes)
                .collect();

            let buffer = storage_buffer(device, "input", bytes.len() as u64);
            queue.write_buffer(&buffer, 0, &bytes);
            (input_binding(module, kind, nth), buffer)
        })
        .collect();

    // The automatic layout only lists the bindings the entry point reads.
    let mut entries = vec![
        wgpu::BindGroupEntry {
            binding: output_binding(module, trace.output_kind()),
            resource: out.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: SAT_BINDING,
            resource: sat_out.as_entire_binding(),
        },
    ];
    for (binding, buffer) in &inputs {
        entries.push(wgpu::BindGroupEntry {
            binding: *binding,
            resource: buffer.as_entire_binding(),
        });
    }
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("trace bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    let out_readback = readback_buffer(device, "out readback", out_bytes);
    let sat_readback = readback_buffer(device, "sat readback", 8);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("trace encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((rows.len() as u32).div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out, 0, &out_readback, 0, out_bytes);
    encoder.copy_buffer_to_buffer(&sat_out, 0, &sat_readback, 0, 8);
    queue.submit([encoder.finish()]);

    let got = read_u32s(device, &out_readback);
    let sat = read_u32s(device, &sat_readback);
    (got, [sat[0], sat[1]])
}

impl Trace {
    // Every operation writes exactly one value plus the saturation count.
    fn output_kind(&self) -> Kind {
        let kinds: Vec<Kind> = self
            .outputs
            .iter()
            .copied()
            .filter(|&k| k != Kind::Sat)
            .collect();
        assert_eq!(kinds.len(), 1, "a trace must carry one result column");
        kinds[0]
    }

    fn output_words(&self) -> usize {
        self.output_kind().words()
    }

    // Renders a case the way the trace file writes it, so a search finds the
    // line in the file.
    fn format_case(&self, row: &[u64]) -> String {
        let fields: Vec<String> = self
            .inputs
            .iter()
            .chain(&self.outputs)
            .zip(row)
            .map(|(&kind, &v)| match kind {
                Kind::Sat => v.to_string(),
                Kind::Q16 | Kind::Mask => format!("{:08x}", v as u32),
                Kind::Q32 | Kind::Q48 => format!("{v:016x}"),
            })
            .collect();
        fields.join(" ")
    }

    fn sat_column(&self) -> usize {
        let offset = self
            .outputs
            .iter()
            .position(|&k| k == Kind::Sat)
            .expect("a trace must carry a sat column");
        self.inputs.len() + offset
    }
}

fn expected_words(trace: &Trace, rows: &[Vec<u64>]) -> Vec<u32> {
    let kind = trace.output_kind();
    let column = trace.inputs.len();
    rows.iter()
        .flat_map(|row| value_words(kind, row[column]))
        .collect()
}

fn expected_saturations(trace: &Trace, rows: &[Vec<u64>]) -> u32 {
    let column = trace.sat_column();
    rows.iter().map(|row| row[column] as u32).sum()
}

fn value_words(kind: Kind, v: u64) -> Vec<u32> {
    match kind.words() {
        1 => vec![v as u32],
        2 => vec![v as u32, (v >> 32) as u32],
        _ => panic!("column {kind:?} has no buffer representation"),
    }
}

// The shader declares one input pair and one destination per width. The n-th
// input of a width goes to the n-th slot of that width.
fn input_binding(module: Module, kind: Kind, nth: usize) -> u32 {
    let slots: &[u32] = match (module, kind) {
        (Module::Batch16, Kind::Q16) => &[1, 2],
        (Module::Batch48, Kind::Q48) => &[1, 2],
        (Module::Batch48, Kind::Q16) => &[5, 6],
        (Module::Batch48, Kind::Q32) => &[8],
        _ => panic!("no input binding for {kind:?}"),
    };
    slots[nth]
}

fn output_binding(module: Module, kind: Kind) -> u32 {
    match (module, kind) {
        (Module::Batch16, Kind::Q16 | Kind::Mask) => 0,
        (Module::Batch48, Kind::Q48) => 0,
        (Module::Batch48, Kind::Q16 | Kind::Mask) => 4,
        (Module::Batch48, Kind::Q32) => 7,
        _ => panic!("no output binding for {kind:?}"),
    }
}

fn source(module: Module, native: bool) -> String {
    let arithmetic = if native { INT64 } else { PORTABLE };
    let batch = match module {
        Module::Batch16 => BATCH16,
        Module::Batch48 => BATCH48,
    };
    format!("{CORE}\n{arithmetic}\n{batch}")
}

fn trace_dir() -> PathBuf {
    std::env::var("FIXED_WGSL_TRACES")
        .unwrap_or_else(|_| "build/traces".to_string())
        .into()
}

fn read_trace(entry_point: &str) -> Trace {
    let path = trace_dir().join(format!("{entry_point}.txt"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!(
            "cannot read {}: {err}\ngenerate the traces with:\n  \
             go run -tags=fixed_satcounter github.com/dhannyell/fixed/cmd/fixedtrace@<tag> -o {}",
            path.display(),
            trace_dir().display(),
        )
    });

    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut rows = Vec::new();

    for line in text.lines() {
        if let Some(header) = line.strip_prefix("# in ") {
            inputs = header.split_whitespace().map(Kind::parse).collect();
            continue;
        }
        if let Some(header) = line.strip_prefix("# out ") {
            outputs = header.split_whitespace().map(Kind::parse).collect();
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }

        let kinds = inputs.iter().chain(&outputs);
        let row: Vec<u64> = line
            .split_whitespace()
            .zip(kinds)
            .map(|(field, &kind)| match kind {
                // The count is a decimal; every other column is raw bits.
                Kind::Sat => field.parse().expect("bad sat count"),
                _ => u64::from_str_radix(field, 16).expect("bad hex field"),
            })
            .collect();

        assert_eq!(
            row.len(),
            inputs.len() + outputs.len(),
            "{}: wrong field count in {line:?}",
            path.display()
        );
        rows.push(row);
    }

    assert!(!rows.is_empty(), "{} carries no cases", path.display());
    Trace {
        inputs,
        outputs,
        rows,
    }
}
