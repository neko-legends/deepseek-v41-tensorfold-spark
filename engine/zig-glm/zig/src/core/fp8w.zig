//! FP8 weights of ModelOpt MIXED_PRECISION checkpoints (GLM-5.3-Flash-FP4-Local) widened to bf16 rows at load:
//!   MXFP8      e4m3 [n, k], scale u8 UE8M0 [n, k/32]: w = code * 2^(s - 127). Exact in bf16 (4 significant bits).
//!   FP8_PB_WO  e4m3 [n, k], scale f32 [n/128, 1, k/128, 1] (ModelOpt's export shape): w = code * s, fp32 product
//!              rounded to bf16 (nearest even), as a bf16 checkpoint of the same weights would hold them.
//! A native FP8 matmul can replace the widening later; the loader's bf16 paths take these as they take bf16 tensors.

const std = @import("std");
const types = @import("quant/types.zig");

const Tensor = types.Tensor;

pub const mx_group = 32;
pub const pb_block = 128;

pub const Format = enum { mxfp8, fp8_pb };

/// Which FP8 layout `w` with `scale` is (by the scale's dtype and shape), else error.UnexpectedTensor.
pub fn format(w: Tensor, scale: Tensor) !Format {
    if (w.dtype != .f8_e4m3 or w.rank != 2) return error.UnexpectedTensor;
    const n = w.shape[0];
    const k = w.shape[1];
    if ((scale.dtype == .u8 or scale.dtype == .f8_e8m0) and scale.rank == 2 and k % mx_group == 0 and
        scale.shape[0] == n and scale.shape[1] == k / mx_group) return .mxfp8;
    if (scale.dtype == .f32 and scale.rank == 4 and scale.shape[1] == 1 and scale.shape[3] == 1 and
        scale.shape[0] == (n + pb_block - 1) / pb_block and scale.shape[2] == (k + pb_block - 1) / pb_block) return .fp8_pb;
    return error.UnexpectedTensor;
}

/// An e4m3 code's value in fp32 (exact); 0x7f / 0xff are NaN.
pub fn e4m3(b: u8) f32 {
    if (b & 0x7f == 0x7f) return std.math.nan(f32);
    const e: i32 = (b >> 3) & 0xf;
    const m: f32 = @floatFromInt(b & 7);
    const v = if (e == 0) std.math.ldexp(m, -9) else std.math.ldexp(8 + m, e - 10);
    return if (b & 0x80 != 0) -v else v;
}

/// fp32 -> bf16 bits, nearest even, NaN kept quiet (torch's float -> bfloat16).
pub fn bf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// Rows [r0, r1) of `w` widened into `out` ((r1 - r0) * k bf16 values), by the scale's format.
pub fn rows(w: Tensor, scale: Tensor, r0: usize, r1: usize, out: []u16) !void {
    const f = try format(w, scale);
    const n = w.shape[0];
    const k = w.shape[1];
    if (r0 > r1 or r1 > n or out.len != (r1 - r0) * k) return error.UnexpectedTensor;
    switch (f) {
        .mxfp8 => for (r0..r1) |r| for (0..k) |c| {
            const s = scale.bytes[r * (k / mx_group) + c / mx_group];
            const v = if (s == 0xff) std.math.nan(f32) else std.math.ldexp(e4m3(w.bytes[r * k + c]), @as(i32, s) - 127);
            out[(r - r0) * k + c] = bf16(v);
        },
        .fp8_pb => {
            const kb = scale.shape[2];
            for (r0..r1) |r| for (0..k) |c| {
                const s = std.mem.bytesToValue(f32, scale.bytes[4 * ((r / pb_block) * kb + c / pb_block) ..][0..4]);
                out[(r - r0) * k + c] = bf16(e4m3(w.bytes[r * k + c]) * s);
            };
        },
    }
}

fn t2(dtype: types.DType, a: usize, b: usize, bytes: []const u8) Tensor {
    return .{ .dtype = dtype, .rank = 2, .shape = .{ a, b, 1, 1, 1 }, .bytes = bytes };
}

test "e4m3 values match the f64 reference for every code" {
    const fp8 = @import("quant/fp8.zig");
    for (0..256) |i| {
        const b: u8 = @intCast(i);
        if (b & 0x7f == 0x7f) {
            try std.testing.expect(std.math.isNan(e4m3(b)));
            continue;
        }
        try std.testing.expectEqual(@as(f32, @floatCast(fp8.e4m3(b))), e4m3(b));
    }
}

test "MXFP8 widens exactly: code * 2^(s - 127)" {
    // one row of 64: two groups, scales 2^0 and 2^-3
    var codes: [64]u8 = undefined;
    for (&codes, 0..) |*c, i| c.* = @intCast(i * 3 % 0x7e);
    codes[5] = 0xc0; // -2
    const sc = [_]u8{ 127, 124 };
    const w = t2(.f8_e4m3, 1, 64, &codes);
    const s = t2(.u8, 1, 2, &sc);
    try std.testing.expectEqual(Format.mxfp8, try format(w, s));
    var out: [64]u16 = undefined;
    try rows(w, s, 0, 1, &out);
    for (codes, 0..) |c, i| {
        const want = e4m3(c) * (if (i < 32) @as(f32, 1) else 0.125);
        const got: f32 = @bitCast(@as(u32, out[i]) << 16);
        try std.testing.expectEqual(want, got); // no rounding: an e4m3 value times a power of two fits bf16
    }
    try std.testing.expectEqual(@as(u16, 0xc000), out[5]);
}

test "FP8_PB_WO: [n/128, 1, k/128, 1] scales, fp32 product to bf16 nearest even" {
    const n = 130;
    const k = 256;
    var codes: [n * k]u8 = @splat(0x38); // 1.0
    codes[129 * k + 200] = 0x40; // 2.0, in block (1, 1)
    const inv = [_]f32{ 0.5, 0.25, 3.0, 1.0 / 3.0 };
    const w = t2(.f8_e4m3, n, k, &codes);
    const s: Tensor = .{ .dtype = .f32, .rank = 4, .shape = .{ 2, 1, 2, 1, 1 }, .bytes = std.mem.sliceAsBytes(&inv) };
    try std.testing.expectEqual(Format.fp8_pb, try format(w, s));
    var out: [2 * k]u16 = undefined;
    try rows(w, s, 128, 130, &out); // rows 128, 129: the second row block
    try std.testing.expectEqual(bf16(3.0), out[0]);
    try std.testing.expectEqual(bf16(1.0 / 3.0), out[130]);
    try std.testing.expectEqual(bf16(2.0 / 3.0), out[k + 200]);
    try std.testing.expectEqual(@as(u16, 0x3f2b), out[k + 200]); // 0x3f2aaaab rounds up to 0x3f2b
    var first: [k]u16 = undefined;
    try rows(w, s, 0, 1, &first);
    try std.testing.expectEqual(bf16(0.5), first[0]);
    try std.testing.expectEqual(bf16(0.25), first[255]);
}

test "shapes that are neither layout are refused" {
    const codes: [64]u8 = @splat(0);
    const w = t2(.f8_e4m3, 2, 32, &codes);
    try std.testing.expectError(error.UnexpectedTensor, format(w, t2(.u8, 2, 2, codes[0..4])));
    try std.testing.expectError(error.UnexpectedTensor, format(t2(.bf16, 2, 16, &codes), t2(.u8, 2, 1, codes[0..2])));
}
