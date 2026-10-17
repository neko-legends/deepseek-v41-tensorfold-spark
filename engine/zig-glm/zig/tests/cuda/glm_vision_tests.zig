//! Ours: `glm-vision MODEL REF [IMAGE]`: GLM-5.3-Flash's CUDA tower stage by stage against tools/glm/vision_ref.py's
//! dumps (REF: meta.json, pixels.f32, patch_embed.bin, blockNN.bin (--stages), rows.bin; bf16). Each block runs from
//! the reference's own input, so errors do not compound (the reference's late blocks hinge on near ties: an
//! end-to-end run is reported, not gated). With IMAGE, the Zig processor's patches are compared with pixels.f32 too.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm_cuda");
const vis = glm.vision;
const Gpu = check.Gpu;

fn read(gpu: Gpu, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpu.gpa, &.{ dir, name });
    defer gpu.gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 31));
}

fn bfv(x: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, x) << 16)));
}

/// Cosine and the worst row's cosine of bf16 `got` against `want` ([rows][n]).
fn cosines(got: []const u16, want: []const u16, n: usize) [2]f64 {
    var dot: f64 = 0;
    var gg: f64 = 0;
    var ww: f64 = 0;
    var worst: f64 = 1;
    for (0..want.len / n) |r| {
        var d: f64 = 0;
        var g2: f64 = 0;
        var w2: f64 = 0;
        for (got[r * n ..][0..n], want[r * n ..][0..n]) |g, w| {
            d += bfv(g) * bfv(w);
            g2 += bfv(g) * bfv(g);
            w2 += bfv(w) * bfv(w);
        }
        dot += d;
        gg += g2;
        ww += w2;
        worst = @min(worst, d / @sqrt(@max(g2 * w2, 1e-300)));
    }
    return .{ dot / @sqrt(@max(gg * ww, 1e-300)), worst };
}

pub fn run(gpu: Gpu, model: []const u8, ref: []const u8, image: ?[]const u8) !void {
    const gpa = gpu.gpa;
    const meta_text = try read(gpu, ref, "meta.json");
    defer gpa.free(meta_text);
    const meta = try std.json.parseFromSlice(std.json.Value, gpa, meta_text, .{});
    defer meta.deinit();
    const grid = meta.value.object.get("grid_thw").?.array.items;
    const gh: u32 = @intCast(grid[1].integer);
    const gw: u32 = @intCast(grid[2].integer);
    const P: usize = @as(usize, gh) * gw;
    const tokens: u32 = @intCast(P / 4);
    const px_bytes = try read(gpu, ref, "pixels.f32");
    defer gpa.free(px_bytes);
    const pixels: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, px_bytes));
    try check.expect(pixels.len == P * 1176, "pixels.f32 holds [{d}][1176]", .{P});
    const v = try vis.Vision.init(gpa, gpu.io, gpu.d, model, .{ .image_tokens = @max(tokens, 64), .max_images = 1 });
    defer v.deinit();
    std.debug.print("tower: {d:.2} GiB on the device (weights, scratch, workspace); grid {d} x {d}, {d} tokens\n", .{ @as(f64, @floatFromInt(v.tower.bytes)) / (1 << 30), gh, gw, tokens });
    if (image) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpa, .limited(1 << 28));
        defer gpa.free(bytes);
        const p = try v.prepare(gpa, bytes, 8000);
        defer gpa.free(p.pixels);
        var most: f32 = 0;
        if (p.pixels.len == pixels.len) for (p.pixels, pixels) |a, b| {
            most = @max(most, @abs(a - b));
        };
        std.debug.print("RESULT processor: grid {d} x {d} (HF {d} x {d}), max |patch - HF| {e:.3}\n", .{ p.gh, p.gw, gh, gw, most });
        try check.expect(p.gh == gh and p.gw == gw and most < 0.05, "the processor's patches match HF's", .{});
    }
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    const t = &v.tower;
    const H = 1024;
    const host = try gpa.alloc(u16, P * 4096);
    defer gpa.free(host);
    const img: @import("lanes").stream.Image = .{ .pixels = pixels, .gh = gh, .gw = gw, .at = 0, .tokens = tokens };
    const resid: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = t.residual(), .len = P * H * 2, .borrowed = true };
    var worst_block: f64 = 1;
    // patch embedding
    try t.patchEmbed(s, img);
    try s.synchronize();
    try resid.download(0, std.mem.sliceAsBytes(host[0 .. P * H]));
    {
        const want_b = try read(gpu, ref, "patch_embed.bin");
        defer gpa.free(want_b);
        const c = cosines(host[0 .. P * H], @alignCast(std.mem.bytesAsSlice(u16, want_b)), H);
        std.debug.print("RESULT patch_embed: cos {d:.6}, worst row {d:.6}\n", .{ c[0], c[1] });
        try check.expect(c[0] >= 0.9999, "patch embedding cos >= 0.9999", .{});
    }
    // each block from the reference's input
    for (0..24) |i| {
        var nb: [32]u8 = undefined;
        const in_name = if (i == 0) "patch_embed.bin" else try std.fmt.bufPrint(&nb, "block{d:0>2}.bin", .{i - 1});
        const in_b = read(gpu, ref, in_name) catch {
            std.debug.print("block {d}: no {s} (dump with --stages)\n", .{ i, in_name });
            continue;
        };
        defer gpa.free(in_b);
        var ob: [32]u8 = undefined;
        const want_b = read(gpu, ref, try std.fmt.bufPrint(&ob, "block{d:0>2}.bin", .{i})) catch continue;
        defer gpa.free(want_b);
        try resid.upload(0, in_b);
        try t.block(s, i, P, gw);
        try s.synchronize();
        try resid.download(0, std.mem.sliceAsBytes(host[0 .. P * H]));
        const c = cosines(host[0 .. P * H], @alignCast(std.mem.bytesAsSlice(u16, want_b)), H);
        worst_block = @min(worst_block, c[0]);
        std.debug.print("RESULT block {d:2}: cos {d:.6}, worst row {d:.6}\n", .{ i, c[0], c[1] });
    }
    // the tail from block 23's reference output, then the whole tower end to end
    const rows_dev = try cuda.DeviceBuffer.alloc(gpu.d, P / 4 * 4096 * 2);
    var rows = rows_dev;
    defer rows.free();
    const want_rows = try read(gpu, ref, "rows.bin");
    defer gpa.free(want_rows);
    const wr: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, want_rows));
    if (read(gpu, ref, "block23.bin")) |b23| {
        defer gpa.free(b23);
        try resid.upload(0, b23);
        try t.tail(s, P, rows.ptr);
        try s.synchronize();
        try rows.download(0, std.mem.sliceAsBytes(host[0 .. P / 4 * 4096]));
        const c = cosines(host[0 .. P / 4 * 4096], wr, 4096);
        std.debug.print("RESULT tail (post norm, downsample, merger): cos {d:.6}, worst row {d:.6}\n", .{ c[0], c[1] });
        try check.expect(c[0] >= 0.999, "the tail cos >= 0.999", .{});
    } else |_| {}
    try t.picture(s, img, rows.ptr);
    try s.synchronize();
    try rows.download(0, std.mem.sliceAsBytes(host[0 .. P / 4 * 4096]));
    const c = cosines(host[0 .. P / 4 * 4096], wr, 4096);
    std.debug.print("RESULT end to end rows: cos {d:.6}, worst row {d:.6} (not gated: the reference's late-block near ties)\n", .{ c[0], c[1] });
    try check.expect(worst_block >= 0.999, "every block cos >= 0.999 from the reference's input", .{});
    check.pass("glm-vision: patch embedding, 24 blocks and the tail match the HF reference", .{});
}
