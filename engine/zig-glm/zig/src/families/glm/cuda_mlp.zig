//! GLM-5.3-Flash's dense MLP (layers 0-2), a rank's share of the Python family's mlp_block: gate/up, SwiGLU, down.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const b16 = @import("cuda_b16.zig");

/// One dense layer on a rank: gate and up [width][dims] and down [dims][width], NVFP4 as Fp4Linear holds them.
pub const Fp4 = struct { gate: cuda.qmmf.Weight, up: cuda.qmmf.Weight, down: cuda.qmmf.Weight };

/// ours: a dense layer stored FP8 (GLM-5.3-Flash-FP4-Local's MXFP8) widened to bf16 as the shared expert holds its
/// weights: [gate | up] [2 width][dims] and down [dims][width], on the B16 matmuls (cuda_moe.fill's shared expert).
pub const B16 = struct { gu: u64, down: u64 };

pub const Layer = struct { fp4: ?Fp4 = null, b16: ?B16 = null, width: usize, dims: usize, limit: f32 };

/// The block's scratch for up to `rows` rows: one projection's rows, the [gate | up] rows, the activation and its sums;
/// ours: the B16 matmuls' split-K partials (the buffers' shared mm_part).
pub const Scratch = struct { tmp: u64, gu: u64, act: u64, xs_act: u64, mm_part: u64 = 0 };

pub const Kernels = struct { set: glue.Set, linear: cuda.qlinear.Linear, copy_rows: cuda.Function };

/// Fp4Linear.__call__ (prefill on a prompt chunk): bf16 out, or fp32 with `wide`; the prompt GEMM needs `ldx` % 8 == 0.
fn mm(k: Kernels, stream: cuda.Stream, w: cuda.qmmf.Weight, x: u64, ldx: usize, rows: usize, out: u64, wide: bool, prompt: bool) !void {
    const q: cuda.qlinear.Weight = .{ .nvfp4 = w };
    const in: cuda.qlinear.Rows = .{ .x = x, .ldx = ldx };
    if (prompt) {
        std.debug.assert(x % 16 == 0 and (rows == 1 or ldx % 8 == 0));
        return if (wide) k.linear.promptF32(stream, q, in, out, rows) else k.linear.prompt(stream, q, in, out, rows);
    }
    return if (wide) k.linear.decodeF32(stream, q, in, out, rows) else k.linear.decode(stream, q, in, out, rows);
}

/// forward.mlp_block for `rows` normed rows (bf16, stride `x_stride`): out [rows][dims] fp32, this rank's partial.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, 0, rows, prompt, out);
}

/// forward.out_proj's fill for the dense MLP: down of rows lo..hi (s.act) into out [hi - lo][dims] fp32.
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    if (l.b16) |w| return b16.matmul(k.set, stream, s.act + lo * l.width * 2, l.width, w.down, out, s.mm_part, hi - lo, l.dims, l.width, true);
    try mm(k, stream, l.fp4.?.down, s.act + lo * l.width * 2, l.width, hi - lo, out, true, prompt);
}

/// The block up to its down projection: [gate | up] and SwiGLU into s.act.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const w = l.width;
    if (l.b16) |b| {
        try b16.matmul(k.set, stream, x, x_stride, b.gu, s.gu, s.mm_part, rows, 2 * w, l.dims, false);
        return glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
    }
    const f = l.fp4.?;
    for ([_]cuda.qmmf.Weight{ f.gate, f.up }, 0..) |proj, j| {
        // Fp4Linear(out=a column slice): the rows computed contiguous, then copied into their half of each row
        try mm(k, stream, proj, x, x_stride, rows, s.tmp, false, prompt);
        var a: cuda.Args = .{};
        a.add(s.tmp);
        a.add(@as(c_longlong, @intCast(w * 2)));
        a.add(s.gu + j * w * 2);
        a.add(@as(c_longlong, @intCast(2 * w * 2)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(w * 2)));
        try cuda.launch.launch(k.copy_rows, .{ .grid = .{ .x = @intCast((rows * w * 2 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
    try glue.swiglu(k.set, stream, s.gu, s.act, s.xs_act, l.limit, rows, w);
}
