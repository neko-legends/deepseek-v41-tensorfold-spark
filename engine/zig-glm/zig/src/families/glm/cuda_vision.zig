//! Ours: GLM-5.3-Flash's image tower on CUDA (HF Glm5NextVisionModel in bf16, transformers 5.20.1), its rows into the
//! prompt. Rank 0 runs the tower (weights widened to bf16 at load: MXFP8 attention exactly, W4A16 NVFP4 MLP and merger
//! as code x scale x scale_2 in fp32, rounded once, as the HF reference dequantizes); every rank holds a pass's image
//! rows (broadcast by rank 0, cuda_follow's `images` step) and splices them over the `<|image|>` rows' embeddings in
//! every hyper-connection copy. The text model is NoPE: image rows need no positions of their own.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lanes = @import("lanes");
const vc = @import("vision_config.zig");
const patches = @import("image_patches.zig");
const vit = cuda.vit;
const Image = lanes.stream.Image;
const Tensor = core.checkpoint.Tensor;
const Allocator = std.mem.Allocator;

pub const Config = vc.Config;

/// What a request may hold: image tokens in all (the rows buffer, the tower's scratch), and images.
pub const Limits = struct { image_tokens: u32, max_images: u32 };

// ---- a pass's image rows (radio/rows.zig's span model) ----

/// An image's placeholder rows [at, at + n) of the prompt, its rows from `row` of the buffer, `image` its index.
pub const Span = struct { at: u32, n: u32, row: u32, image: u32 };

/// The images a prompt pass from `from` reads, in prompt order, their rows back to back from row 0.
pub fn plan(a: Allocator, images: []const Image, from: u32) ![]Span {
    var out: std.ArrayList(Span) = .empty;
    errdefer out.deinit(a);
    var row: u32 = 0;
    var end: u32 = 0;
    for (images, 0..) |img, i| {
        if (img.at < end or img.tokens == 0) return error.ImageSpans; // in order, apart, each with rows
        end = img.at + img.tokens;
        if (end <= from) continue; // a restored prefix holds these rows' states already
        try out.append(a, .{ .at = img.at, .n = img.tokens, .row = row, .image = @intCast(i) });
        row += img.tokens;
    }
    return out.toOwnedSlice(a);
}

/// Rows the spans hold in all.
pub fn total(spans: []const Span) u32 {
    var n: u32 = 0;
    for (spans) |s| n += s.n;
    return n;
}

/// The part of `sp` prompt rows [lo, hi) cross: its first prompt row, its first buffer row, its rows; null for none.
pub fn cut(sp: Span, lo: u32, hi: u32) ?[3]u32 {
    const from = @max(lo, sp.at);
    const to = @min(hi, sp.at + sp.n);
    return if (from < to) .{ from, sp.row + from - sp.at, to - from } else null;
}

/// Every rank's: the rows buffer (bf16 [limit, hidden]), the pass's spans and slot, the splice kernel.
pub const Images = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    mod: cuda.Module,
    splice_fn: cuda.Function,
    buf: cuda.DeviceBuffer,
    limit: u32,
    hidden: usize,
    spans: std.ArrayList(Span) = .empty,
    slot: u32 = 0,
    active: bool = false,

    pub fn init(gpa: Allocator, d: *const cuda.Driver, limit: u32, hidden: usize) !*Images {
        var mod = try cuda.Module.load(d, cuda.kernels.vit);
        errdefer mod.unload();
        const f = try mod.function("tf_glmv_splice");
        var buf = try cuda.DeviceBuffer.alloc(d, @as(usize, limit) * hidden * 2);
        errdefer buf.free();
        const im = try gpa.create(Images);
        im.* = .{ .gpa = gpa, .d = d, .mod = mod, .splice_fn = f, .buf = buf, .limit = limit, .hidden = hidden };
        return im;
    }

    pub fn deinit(im: *Images) void {
        im.spans.deinit(im.gpa);
        im.buf.free();
        im.mod.unload();
        im.gpa.destroy(im);
    }

    /// Where span `s`'s rows start.
    pub fn at(im: *const Images, s: Span) u64 {
        return im.buf.ptr + @as(u64, s.row) * im.hidden * 2;
    }

    /// The bytes the spans' rows take from the buffer's start (one broadcast).
    pub fn bytes(im: *const Images, spans: []const Span) usize {
        return @as(usize, total(spans)) * im.hidden * 2;
    }

    /// A pass on `slot` reads `spans` (their rows in the buffer); refused past the buffer.
    pub fn set(im: *Images, spans: []const Span, slot: u32) !void {
        for (spans) |sp| if (sp.row + sp.n > im.limit) return error.ImageRows;
        im.spans.clearRetainingCapacity();
        try im.spans.appendSlice(im.gpa, spans);
        im.slot = slot;
        im.active = spans.len > 0;
    }

    pub fn clear(im: *Images) void {
        im.spans.clearRetainingCapacity();
        im.active = false;
    }

    /// The pass's splice for `slot`'s sequence, or null (no images, or another stream's).
    pub fn splice(im: *const Images, slot: u32) ?Splice {
        if (!im.active or im.slot != slot) return null;
        return .{ .im = im };
    }
};

/// Prompt rows [lo, lo + R) embedded at `x` (bf16 [R][copies][hidden]) take the image rows they cross.
pub const Splice = struct {
    im: *const Images,

    pub fn run(sp: Splice, s: cuda.Stream, lo: usize, R: usize, x: u64, copies: usize) !void {
        const im = sp.im;
        const w = im.hidden * 2;
        for (im.spans.items) |span| {
            const from, const row, const n = cut(span, @intCast(lo), @intCast(lo + R)) orelse continue;
            var a: cuda.Args = .{};
            a.add(im.buf.ptr + @as(u64, row) * w);
            a.add(x + @as(u64, from - lo) * w * copies);
            a.add(@as(c_int, @intCast(im.hidden)));
            a.add(@as(c_int, @intCast(copies)));
            try cuda.launch.launch(im.splice_fn, .{ .grid = .{ .x = n, .y = @intCast((im.hidden + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
        }
    }
};

// ---- the tower (rank 0) ----

const Block = struct { norm1: u64, norm2: u64, qkv: u64, qkv_b: u64, qn: u64, kn: u64, proj: u64, proj_b: u64, gu: u64, gu_b: u64, down: u64, down_b: u64 };

const Weights = struct {
    patch: u64 = 0,
    patch_b: u64 = 0,
    blocks: [vc.depth]Block = undefined,
    post: u64 = 0,
    ds: u64 = 0, // downsample [4096, 2, 2, 1024]: a merge window's four patch rows side by side
    ds_b: u64 = 0,
    m_proj: u64 = 0,
    m_ln_w: u64 = 0,
    m_ln_b: u64 = 0,
    m_gu: u64 = 0, // [gate | up] [20480, 4096]
    m_down: u64 = 0,
    inv: u64 = 0, // f32 [16]: the rope's inverse frequencies
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,

    fn deinit(w: *Weights, gpa: Allocator) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(gpa);
    }
};

/// The vision tensors as bf16 on the device, named under model.visual.
const Loader = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    w: *Weights,
    host: std.ArrayList(u16) = .empty,
    name: [160]u8 = undefined,

    fn deinit(L: *Loader) void {
        L.host.deinit(L.gpa);
    }

    fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !Tensor {
        return L.ck.get(try std.fmt.bufPrint(&L.name, "model.visual." ++ fmt, args));
    }

    fn put(L: *Loader, values: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(L.d, values);
        try L.w.buffers.append(L.gpa, b);
        L.w.bytes += b.len;
        return b.ptr;
    }

    /// The tensor's elements appended to `host` as bf16: bf16 as stored, MXFP8 widened exactly, NVFP4 dequantized.
    fn append(L: *Loader, comptime fmt: []const u8, args: anytype) !void {
        var buf: [160]u8 = undefined;
        const base = try std.fmt.bufPrint(&buf, "model.visual." ++ fmt, args);
        const t = try L.ck.get(base);
        switch (t.dtype) {
            .bf16 => try L.host.appendSlice(L.gpa, @alignCast(std.mem.bytesAsSlice(u16, t.bytes))),
            .f8_e4m3 => { // MXFP8 (attention): e4m3 x 2^(ue8m0 - 127) per 32 inputs, exact in bf16
                var sb: [168]u8 = undefined;
                const s = try L.ck.get(try std.fmt.bufPrint(&sb, "{s}_scale", .{base}));
                const at = L.host.items.len;
                try L.host.resize(L.gpa, at + t.numel());
                try core.fp8w.rows(t, s, 0, t.dim(0), L.host.items[at..]);
            },
            .u8 => { // W4A16 NVFP4 (MLP, merger): e2m1 (input 2j in byte j's low nibble) x e4m3 per 16 x scale_2
                if (!std.mem.endsWith(u8, base, ".weight")) return error.UnexpectedTensor;
                const module = base[0 .. base.len - ".weight".len];
                var sb: [176]u8 = undefined;
                const s = try L.ck.get(try std.fmt.bufPrint(&sb, "{s}.weight_scale", .{module}));
                const g = try L.ck.get(try std.fmt.bufPrint(&sb, "{s}.weight_scale_2", .{module}));
                const n = t.dim(0);
                const k = 2 * t.dim(1);
                if (t.rank != 2 or !s.is(.f8_e4m3, &.{ n, k / 16 }) or g.dtype != .f32 or g.numel() != 1) return error.BadNvfp4;
                const global = std.mem.bytesToValue(f32, g.bytes[0..4]);
                const at = L.host.items.len;
                try L.host.resize(L.gpa, at + n * k);
                const out = L.host.items[at..];
                for (0..n) |r| for (0..k) |c| {
                    const byte = t.bytes[r * k / 2 + c / 2];
                    const code: u8 = if (c % 2 == 0) byte & 0xf else byte >> 4;
                    const v: f32 = @as(f32, @floatCast(core.quant.nvfp4.e2m1(code))) * core.fp8w.e4m3(s.bytes[r * k / 16 + c / 16]);
                    out[r * k + c] = core.fp8w.bf16(v * global);
                };
            },
            else => return error.UnexpectedTensor,
        }
    }

    /// The tensors named, appended in order, uploaded as one.
    fn one(L: *Loader, comptime fmt: []const u8, args: anytype) !u64 {
        L.host.clearRetainingCapacity();
        try L.append(fmt, args);
        return L.put(std.mem.sliceAsBytes(L.host.items));
    }

    /// [gate | up] rows (or biases) as one tensor: a Linear whose row holds the gate's outputs, then the up's.
    fn pair(L: *Loader, comptime fmt: []const u8, args: anytype, comptime what: []const u8) !u64 {
        L.host.clearRetainingCapacity();
        try L.append(fmt ++ ".gate_proj." ++ what, args);
        try L.append(fmt ++ ".up_proj." ++ what, args);
        return L.put(std.mem.sliceAsBytes(L.host.items));
    }
};

fn loadWeights(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !Weights {
    var w: Weights = .{};
    errdefer w.deinit(gpa);
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, "model.visual.");
    defer ck.close();
    var L: Loader = .{ .gpa = gpa, .d = d, .ck = &ck, .w = &w };
    defer L.deinit();
    w.patch = try L.one("patch_embed.proj.weight", .{}); // [1024, 3, 2, 14, 14]: the patch layout's (c, t, y, x)
    w.patch_b = try L.one("patch_embed.proj.bias", .{});
    for (&w.blocks, 0..) |*b, i| {
        b.norm1 = try L.one("blocks.{d}.norm1.weight", .{i});
        b.norm2 = try L.one("blocks.{d}.norm2.weight", .{i});
        b.qkv = try L.one("blocks.{d}.attn.qkv.weight", .{i});
        b.qkv_b = try L.one("blocks.{d}.attn.qkv.bias", .{i});
        b.qn = try L.one("blocks.{d}.attn.q_norm.weight", .{i});
        b.kn = try L.one("blocks.{d}.attn.k_norm.weight", .{i});
        b.proj = try L.one("blocks.{d}.attn.proj.weight", .{i});
        b.proj_b = try L.one("blocks.{d}.attn.proj.bias", .{i});
        b.gu = try L.pair("blocks.{d}.mlp", .{i}, "weight");
        b.gu_b = try L.pair("blocks.{d}.mlp", .{i}, "bias");
        b.down = try L.one("blocks.{d}.mlp.down_proj.weight", .{i});
        b.down_b = try L.one("blocks.{d}.mlp.down_proj.bias", .{i});
        ck.release();
    }
    w.post = try L.one("post_layernorm.weight", .{});
    // downsample [O, C, 2, 2] -> [O, 2, 2, C] (vision.zig's permute): a window's four patch rows, tap by tap
    {
        L.host.clearRetainingCapacity();
        try L.append("downsample.weight", .{});
        const H = vc.hidden;
        const tmp = try gpa.alloc(u16, H * 4);
        defer gpa.free(tmp);
        if (L.host.items.len != vc.out_hidden * H * 4) return error.VisionShape;
        for (0..vc.out_hidden) |o| {
            const row = L.host.items[o * H * 4 ..][0 .. H * 4];
            @memcpy(tmp, row);
            for (0..H) |ch| for (0..4) |tap| {
                row[tap * H + ch] = tmp[ch * 4 + tap];
            };
        }
        w.ds = try L.put(std.mem.sliceAsBytes(L.host.items));
    }
    w.ds_b = try L.one("downsample.bias", .{});
    w.m_proj = try L.one("merger.proj.weight", .{});
    w.m_ln_w = try L.one("merger.post_projection_norm.weight", .{});
    w.m_ln_b = try L.one("merger.post_projection_norm.bias", .{});
    w.m_gu = try L.pair("merger", .{}, "weight");
    w.m_down = try L.one("merger.down_proj.weight", .{});
    // HF's axial rope: inv_freq = 1 / 10000^(arange(0, 32, 2) / 32) in fp32
    var inv: [16]f32 = undefined;
    for (&inv, 0..) |*f, k| f.* = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * k)) / 32.0);
    w.inv = try L.put(std.mem.asBytes(&inv));
    return w;
}

/// The tower's own kernels (vit.cu, ours: tf_glmv_*) beside the shared vision glue.
const Kernels = struct {
    mod: cuda.Module,
    rms: cuda.Function,
    qk_rope: cuda.Function,
    swiglu: cuda.Function,
    layer_norm: cuda.Function,

    fn load(d: *const cuda.Driver) !Kernels {
        var mod = try cuda.Module.load(d, cuda.kernels.vit);
        errdefer mod.unload();
        return .{ .mod = mod, .rms = try mod.function("tf_glmv_rms"), .qk_rope = try mod.function("tf_glmv_qk_rope"), .swiglu = try mod.function("tf_glmv_swiglu"), .layer_norm = try mod.function("tf_glmv_layer_norm") };
    }
};

fn go(f: cuda.Function, s: cuda.Stream, x: usize, y: usize, a: *cuda.Args) !void {
    if (x == 0) return;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(x), .y = @intCast(y) }, .block = .{ .x = 256 } }, s, a);
}

/// Rows a Linear whose bias cuBLASLt cannot fuse sums through f32 at once.
const acc_rows = 4096;

pub const Tower = struct {
    gpa: Allocator,
    c: Config,
    w: Weights,
    k: Kernels,
    vk: vit.Kernels,
    at: vit.Attention,
    g: vit.Gemms,
    scratch: cuda.DeviceBuffer,
    b: Buffers,
    max_patches: usize,
    bytes: u64, // on the device: weights, scratch and the Linears' workspace

    const Buffers = struct { px32: u64, px: u64, h: u64, x: u64, wide: u64, att: u64, act: u64, acc: u64 };

    /// The tower from `dir` and scratch for an image of `image_tokens` tokens (4 patches a token).
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, c: Config, image_tokens: u32) !Tower {
        var w = try loadWeights(gpa, io, d, dir);
        errdefer w.deinit(gpa);
        var k = try Kernels.load(d);
        errdefer k.mod.unload();
        var vk = try vit.Kernels.load(d);
        errdefer vk.unload();
        var at = try vit.Attention.load(d);
        errdefer at.unload();
        var g = try vit.Gemms.init(gpa, d);
        errdefer g.deinit();
        const P: usize = @as(usize, image_tokens) * vc.patches_per_token;
        const H = vc.hidden;
        const wide = @max(2 * vc.mlp_inter, 3 * H) * P; // elements: qkv, the MLP's [gate | up], the merger's (P / 4 x 20480)
        const sizes = [_]usize{ 4 * P * vc.patch_width, 2 * P * vc.patch_width, 2 * P * H, 2 * P * H, 2 * wide, 2 * P * H, 2 * P * vc.mlp_inter, 4 * acc_rows * 2 * vc.merger_inter };
        var sum: usize = 0;
        for (sizes) |sz| sum += std.mem.alignForward(usize, sz, 256);
        var scratch = try cuda.DeviceBuffer.alloc(d, sum);
        errdefer scratch.free();
        var p: [sizes.len]u64 = undefined;
        var off: usize = 0;
        for (sizes, &p) |sz, *ptr| {
            ptr.* = try scratch.at(off);
            off += std.mem.alignForward(usize, sz, 256);
        }
        return .{ .gpa = gpa, .c = c, .w = w, .k = k, .vk = vk, .at = at, .g = g, .scratch = scratch, .max_patches = P, .b = .{ .px32 = p[0], .px = p[1], .h = p[2], .x = p[3], .wide = p[4], .att = p[5], .act = p[6], .acc = p[7] }, .bytes = w.bytes + sum + g.workspace.len };
    }

    pub fn deinit(t: *Tower) void {
        t.scratch.free();
        t.g.deinit();
        t.at.unload();
        t.vk.unload();
        t.k.mod.unload();
        t.w.deinit(t.gpa);
    }

    /// y [rows, n] = bf16(x . w^T + b): the bias in cuBLASLt's epilogue, else through f32 a block of rows at a time.
    fn linear(t: *Tower, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, n: usize, kk: usize) !void {
        if (try t.g.fused(rows, n, kk)) return vit.linear(&t.g, &t.vk, s, t.b.acc, x, w, b, y, rows, n, kk);
        var r0: usize = 0;
        while (r0 < rows) : (r0 += acc_rows) {
            const r = @min(acc_rows, rows - r0);
            try t.g.run(s, x + r0 * kk * 2, w, t.b.acc, r, n, kk, .f32);
            try t.vk.bias(s, t.b.acc, b, y + r0 * n * 2, r, n);
        }
    }

    fn rms(t: *Tower, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ x, w, y }) |p| a.add(p);
        a.add(@as(c_int, vc.hidden));
        a.add(t.c.eps);
        try go(t.k.rms, s, rows, 1, &a);
    }

    fn swiglu(t: *Tower, s: cuda.Stream, gu: u64, out: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(gu);
        a.add(out);
        a.add(@as(i64, @intCast(rows)));
        a.add(@as(c_int, @intCast(n)));
        a.add(t.c.limit);
        try go(t.k.swiglu, s, (rows * n + 255) / 256, 1, &a);
    }

    /// Each span's image through the tower into its rows of `im` (bf16 [tokens, 4096]).
    pub fn encode(t: *Tower, s: cuda.Stream, images: []const Image, spans: []const Span, im: *const Images) !void {
        if (im.hidden != vc.out_hidden) return error.RowShape;
        for (spans) |sp| {
            const img = images[sp.image];
            if (img.tokens != sp.n or sp.row + sp.n > im.limit) return error.ImageRows;
            try t.picture(s, img, im.at(sp));
        }
    }

    /// One image's rows at `out`: patches, 24 blocks, post norm, downsample, merger.
    pub fn picture(t: *Tower, s: cuda.Stream, img: Image, out: u64) !void {
        try t.patchEmbed(s, img);
        const P: usize = @as(usize, img.gh) * img.gw;
        for (0..vc.depth) |i| try t.block(s, i, P, img.gw);
        try t.tail(s, P, out);
    }

    /// The residual stream (bf16 [P, 1024]): after patchEmbed, each block's output (the checks upload into it).
    pub fn residual(t: *const Tower) u64 {
        return t.b.h;
    }

    /// The patches (f32, uploaded and cast) through the patch embedding into the residual stream.
    pub fn patchEmbed(t: *Tower, s: cuda.Stream, img: Image) !void {
        const b = t.b;
        const P: usize = @as(usize, img.gh) * img.gw;
        if (P == 0 or P > t.max_patches or img.gh % 2 != 0 or img.gw % 2 != 0 or P / 4 != img.tokens or img.pixels.len != P * vc.patch_width) return error.ImageShape;
        try t.scratch.uploadAsync(b.px32 - t.scratch.ptr, std.mem.sliceAsBytes(img.pixels), s.handle);
        try t.vk.cast(s, b.px32, b.px, P * vc.patch_width);
        try t.linear(s, b.px, t.w.patch, t.w.patch_b, b.h, P, vc.hidden, vc.patch_width); // Conv3d k = stride: a Linear
    }

    /// Block `i` over the residual stream of P patches of a grid `gw` wide.
    pub fn block(t: *Tower, s: cuda.Stream, i: usize, P: usize, gw: usize) !void {
        const b = t.b;
        const H = vc.hidden;
        const l = &t.w.blocks[i];
        const scale: f32 = 1.0 / @sqrt(@as(f32, vc.head_dim));
        try t.rms(s, b.h, l.norm1, b.x, P);
        try t.linear(s, b.x, l.qkv, l.qkv_b, b.wide, P, 3 * H, H);
        var a: cuda.Args = .{};
        for ([_]u64{ b.wide, l.qn, l.kn, t.w.inv }) |p| a.add(p);
        a.add(@as(c_int, @intCast(gw)));
        a.add(t.c.eps);
        try go(t.k.qk_rope, s, P, 1, &a);
        try t.at.full(s, b.wide, b.att, P, vc.heads, vc.head_dim, scale);
        try t.linear(s, b.att, l.proj, l.proj_b, b.x, P, H, H);
        try t.vk.pointwise(s, b.x, b.h, b.h, P * H, .none);
        try t.rms(s, b.h, l.norm2, b.x, P);
        try t.linear(s, b.x, l.gu, l.gu_b, b.wide, P, 2 * vc.mlp_inter, H);
        try t.swiglu(s, b.wide, b.act, P, vc.mlp_inter);
        try t.linear(s, b.act, l.down, l.down_b, b.x, P, H, vc.mlp_inter);
        try t.vk.pointwise(s, b.x, b.h, b.h, P * H, .none);
    }

    /// The post norm, the 2x2 downsample and the merger: the residual stream to the image's rows at `out`.
    pub fn tail(t: *Tower, s: cuda.Stream, P: usize, out: u64) !void {
        const b = t.b;
        const w = &t.w;
        const H = vc.hidden;
        try t.rms(s, b.h, w.post, b.x, P);
        // [P, 1024] is [P / 4, 4 x 1024]: the patches are block-major, a window's four rows in a row
        const T = P / 4;
        const O = vc.out_hidden;
        try t.linear(s, b.x, w.ds, w.ds_b, b.att, T, O, 4 * H);
        try t.g.run(s, b.att, w.m_proj, b.x, T, O, O, .bf16);
        var a: cuda.Args = .{};
        for ([_]u64{ b.x, w.m_ln_w, w.m_ln_b, b.att }) |p| a.add(p);
        a.add(@as(c_int, O));
        a.add(@as(f32, 1e-5)); // nn.LayerNorm's default
        try go(t.k.layer_norm, s, T, 1, &a);
        try t.vk.pointwise(s, b.att, null, b.att, T * O, .gelu);
        try t.g.run(s, b.att, w.m_gu, b.wide, T, 2 * vc.merger_inter, O, .bf16);
        try t.swiglu(s, b.wide, b.act, T, vc.merger_inter);
        try t.g.run(s, b.act, w.m_down, out, T, O, vc.merger_inter, .bf16);
    }
};

/// Rank 0's image side: the tower and what the server's offer reads (the processor's config, markers, limits).
pub const Vision = struct {
    gpa: Allocator,
    io: std.Io,
    c: Config,
    tower: Tower,
    limits: Limits,
    markers: [3]u32, // <|begin_of_image|>, <|image|>, <|end_of_image|>: text that spells one stays text

    /// The checkpoint's tower; error.NoVisionTower without a vision_config.
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, limits: Limits) !*Vision {
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 26));
        const proc = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "processor_config.json" }), a, .limited(1 << 20)) catch null;
        const c = (try Config.parse(a, cfg_text, proc)) orelse return error.NoVisionTower;
        var tower = try Tower.init(gpa, io, d, dir, c, limits.image_tokens);
        errdefer tower.deinit();
        const v = try gpa.create(Vision);
        v.* = .{ .gpa = gpa, .io = io, .c = c, .tower = tower, .limits = limits, .markers = .{ c.image_start, c.image_token, c.image_end } };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    /// An image's bytes as the tower reads them, at most `max_tokens` tokens (the server's prepare).
    pub fn prepare(v: *const Vision, a: Allocator, data: []const u8, max_tokens: u32) !patches.Prepared {
        return patches.Processor(core.picture).prepare(v.io, a, data, v.c, max_tokens);
    }

    /// The words for a prepare error the server cannot word (a JPEG library this machine lacks).
    pub const explain = core.picture.decoderProblem;
};

test "a pass from a restored point plans only the images that end past it; a chunk splices the rows it crosses" {
    const a = std.testing.allocator;
    const imgs = [_]Image{
        .{ .pixels = &.{}, .gh = 32, .gw = 32, .at = 5, .tokens = 256 },
        .{ .pixels = &.{}, .gh = 40, .gw = 30, .at = 300, .tokens = 300 },
    };
    const all = try plan(a, &imgs, 0);
    defer a.free(all);
    try std.testing.expectEqualSlices(Span, &.{ .{ .at = 5, .n = 256, .row = 0, .image = 0 }, .{ .at = 300, .n = 300, .row = 256, .image = 1 } }, all);
    const later = try plan(a, &imgs, 400);
    defer a.free(later);
    try std.testing.expectEqualSlices(Span, &.{.{ .at = 300, .n = 300, .row = 0, .image = 1 }}, later);
    try std.testing.expectEqual(@as(u32, 556), total(all));
    // a 2048-row chunk from 256: image 0's rows 251.. and image 1 whole
    try std.testing.expectEqual(@as(?[3]u32, .{ 256, 251, 5 }), cut(all[0], 256, 2304));
    try std.testing.expectEqual(@as(?[3]u32, .{ 300, 256, 300 }), cut(all[1], 256, 2304));
    try std.testing.expectEqual(@as(?[3]u32, null), cut(all[1], 0, 300));
    try std.testing.expectError(error.ImageSpans, plan(a, &.{ imgs[1], imgs[0] }, 0));
}

test "the tower and its loader are analyzed on the host" {
    _ = &Vision.init;
    _ = &Tower.picture;
    _ = &Splice.run;
}
