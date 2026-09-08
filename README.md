# FIXED-WGSL

This library mirrors the original Go [implementation](https://github.com/dhannyell/fixed), incorporating the necessary changes to ensure proper functionality and improved performance in WGSL while maintaining the same output as the CPU version.

## Run Benchmarks

`cargo test --release --test bench -- --ignored --nocapture`

## Run Tests

`cargo test --test q16_gpu`
`cargo test --test q48_gpu`

## License

[MIT](LICENSE)
