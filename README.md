# FIXED-WGSL

This library mirrors the original Go [implementation](https://github.com/dhannyell/fixed), incorporating the necessary changes to ensure proper functionality and improved performance in WGSL while maintaining the same output as the CPU version.

## Generate files and validate

### Portable

```bash
cat fixed_core.wgsl fixed_portable.wgsl batch16.wgsl > build/portable.wgsl && naga build/portable.wgsl
```

It uses only i32 variables and is compatible with all environments where wgpu is supported.

### Native

```bash
cat fixed_core.wgsl fixed_int64.wgsl batch16.wgsl > build/native.wgsl && naga build/native.wgsl
```

It uses 64-bit integer variables and is not browser-compatible; the GPU must support 64-bit variables. In wgpu-native, you need to request the `SHADER_INT64` feature to request or check for support. It is generally faster than the portable version.

On entry-level or integrated GPUs, the native version is generally faster because the bottleneck is pure arithmetic. On intermediate or high-end GPUs, the native version usually performs equal to or at least better than the portable version because the bottleneck becomes bandwidth.

## Run Benchmarks

`cargo test --release --test bench -- --ignored --nocapture`

## Run Tests

`cargo test --test q16_gpu`
`cargo test --test q48_gpu`

## License

[MIT](LICENSE)
