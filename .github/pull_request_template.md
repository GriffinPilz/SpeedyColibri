## Summary

Describe the problem and the smallest change that solves it.

## Validation

- [ ] `cargo test --workspace` passes
- [ ] CUDA changes built + tested on a CUDA host (`cargo test -p colibri-backend --features cuda`)
- [ ] Performance claims include hardware, commands, and repeatable measurements

## Compatibility

- [ ] The default CPU build remains dependency-free
- [ ] No model files, generated binaries, or benchmark artifacts are included
