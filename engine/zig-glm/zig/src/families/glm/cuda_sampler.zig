//! Sampled draws on upstream's sample.cu (#542 option C): every rank runs tf_draw.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const Span = @import("cuda_shard.zig").Span;
const Exchange = @import("cuda_forward.zig").Exchange;

pub const Sampling = lanes.Sampling;

/// The request's rule as the engine runs it: null (or temperature 0) decodes greedily.
pub fn check(s: ?Sampling) ?Sampling {
    const x = s orelse return null;
    return if (x.temperature > 0) x else null;
}

/// sample.cu's Rule: the seed, 1 / T, top_p, the candidates' near window, ln(min_p) and top_k (0: off).
pub const Rule = extern struct { seed: u64, inv_t: f32, top_p: f32, near: f32, min_log: f32, top_k: u32, pad: u32 = 0 };

/// The rule as nemotron/cuda_sampler.zig fills it.
pub fn rule(s: Sampling) Rule {
    return .{ .seed = s.seed, .inv_t = @floatCast(1.0 / @max(s.temperature, 1e-6)), .top_p = @floatCast(s.top_p), .near = 20.0, .min_log = @floatCast(s.minLog()), .top_k = s.top_k };
}

/// A rank's logits rows: bf16 [rows][its vocabulary span].
pub const Local = struct { rank: usize, logits: u64 };

pub const Sampler = struct {
    d: *const cuda.Driver,
    module: cuda.Module,
    tf_draw: cuda.Function,
    tf_grammar_mask: cuda.Function, // structured output (ours): a row's disallowed tokens to -inf before its draw
    vocab: usize,
    rows: usize,
    full: cuda.DeviceBuffer, // bf16 [rows][vocab]: every rank's columns in place
    rule: cuda.DeviceBuffer,
    meta: cuda.DeviceBuffer,
    tok: cuda.DeviceBuffer,
    prob: cuda.DeviceBuffer,
    bits: ?cuda.DeviceBuffer = null, // u32 [rows][words]: the masked rows' allowed-token bits, made at the first

    pub fn init(d: *const cuda.Driver, vocab: usize, rows: usize) !Sampler {
        var m = try cuda.Module.load(d, cuda.kernels.sample);
        errdefer m.unload();
        const f = try m.function("tf_draw");
        const gm = try m.function("tf_grammar_mask");
        var full = try cuda.DeviceBuffer.alloc(d, rows * vocab * 2);
        errdefer full.free();
        var r = try cuda.DeviceBuffer.alloc(d, @sizeOf(Rule));
        errdefer r.free();
        var meta = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer meta.free();
        var tok = try cuda.DeviceBuffer.alloc(d, rows * 4);
        errdefer tok.free();
        const prob = try cuda.DeviceBuffer.alloc(d, rows * 4);
        return .{ .d = d, .module = m, .tf_draw = f, .tf_grammar_mask = gm, .vocab = vocab, .rows = rows, .full = full, .rule = r, .meta = meta, .tok = tok, .prob = prob };
    }

    pub fn deinit(s: *Sampler) void {
        if (s.bits) |*b| b.free();
        s.prob.free();
        s.tok.free();
        s.meta.free();
        s.rule.free();
        s.full.free();
        s.module.unload();
    }

    /// `rows` rows of `mine` -> `out`, tokens at positions first, first + 1, ...; `spans`: every rank's, rank order.
    pub fn draw(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize, first: u64, r: Rule, out: []u32) !void {
        try s.gather(stream, exchange, spans, mine, rows);
        try s.pick(stream, rows, first, r, out);
    }

    /// Every rank's columns of `rows` rows into `full` on every rank (a collective: all ranks call it together).
    pub fn gather(s: *Sampler, stream: cuda.Stream, exchange: Exchange, spans: []const Span, mine: []const Local, rows: usize) !void {
        std.debug.assert(rows <= s.rows);
        const V = s.vocab;
        for (mine) |l| {
            const span = spans[l.rank];
            for (0..rows) |row| try s.copy(stream, s.full.ptr + (row * V + span.lo) * 2, l.logits + row * span.len() * 2, span.len() * 2);
        }
        switch (exchange) {
            .local => {},
            .nccl => |x| { // each rank's columns of each row from that rank (spans may differ in width: TP=3)
                try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                for (0..rows) |row| for (spans, 0..) |span, q| {
                    const at = s.full.ptr + (row * V + span.lo) * 2;
                    try x.lib.check(x.lib.api.ncclBroadcast(at, at, span.len(), .bf16, @intCast(q), x.comm, stream.handle), "ncclBroadcast");
                };
                try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
            },
        }
    }

    /// `full`'s first `rows` rows drawn under `r` (local: no other rank takes part).
    fn pick(s: *Sampler, stream: cuda.Stream, rows: usize, first: u64, r: Rule, out: []u32) !void {
        std.debug.assert(rows <= s.rows and out.len >= rows and first >= 1);
        try s.rule.uploadAsync(0, std.mem.asBytes(&r), stream.handle);
        const meta = [4]i32{ @intCast(first - 1), 0, 0, 0 }; // row r keys meta[0] + r + 1
        try s.meta.uploadAsync(0, std.mem.asBytes(&meta), stream.handle);
        var args: cuda.Args = .{};
        args.add(s.full.ptr);
        args.add(@as(u32, @intCast(s.vocab)));
        args.add(s.rule.ptr);
        args.add(s.meta.ptr);
        args.add(@as(i32, 0));
        args.add(s.tok.ptr);
        args.add(s.prob.ptr);
        try cuda.launch.launch(s.tf_draw, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, stream, &args);
        try stream.synchronize();
        try s.tok.download(0, std.mem.sliceAsBytes(out[0..rows]));
    }

    /// Ours, structured output after `gather` (rank 0 alone): `bits` (u32 [rows][words], empty: none) make the gathered
    /// rows' disallowed tokens -inf, then `r` draws them, or with no rule (greedy) the host picks each row's largest
    /// allowed value, lowest id first among equals (`bestAllowed`, the greedy draw's rule); `host` holds the rows.
    pub fn masked(s: *Sampler, stream: cuda.Stream, rows: usize, first: u64, r: ?Rule, bits: []const u32, host: []u16, out: []u32) !void {
        std.debug.assert(rows <= s.rows and out.len >= rows);
        const V = s.vocab;
        const words = (V + 31) / 32;
        if (bits.len > 0) {
            if (bits.len % rows != 0 or bits.len / rows > words) return error.InvalidGrammarMasks;
            const W = bits.len / rows;
            if (s.bits == null) s.bits = try cuda.DeviceBuffer.alloc(s.d, s.rows * words * 4);
            try s.bits.?.uploadAsync(0, std.mem.sliceAsBytes(bits), stream.handle);
            var args: cuda.Args = .{};
            args.add(s.full.ptr);
            args.add(@as(u32, @intCast(V)));
            args.add(s.bits.?.ptr);
            args.add(@as(u32, @intCast(W)));
            try cuda.launch.launch(s.tf_grammar_mask, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &args);
        }
        if (r) |rule_| return s.pick(stream, rows, first, rule_, out);
        if (host.len < rows * V) return error.WindowTooWide;
        try stream.synchronize();
        try s.rowsRead(rows, host);
        for (0..rows) |row| out[row] = bestAllowed(host[row * V ..][0..V], &.{}) orelse return error.NoAllowedToken;
    }

    /// The whole rows a draw read (bf16 [rows][vocab]), for checks against the host rule.
    pub fn rowsRead(s: *const Sampler, rows: usize, host: []u16) !void {
        try s.full.download(0, std.mem.sliceAsBytes(host[0 .. rows * s.vocab]));
    }

    fn copy(s: *const Sampler, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }
};

/// A row's largest bf16 value among `bits`' allowed tokens (empty: all; NaN and -inf never), the lowest id on ties.
pub fn bestAllowed(row: []const u16, bits: []const u32) ?u32 {
    var best: ?u32 = null;
    var value: f32 = -std.math.inf(f32);
    for (row, 0..) |b, t| {
        if (bits.len > 0 and (t / 32 >= bits.len or (bits[t / 32] >> @intCast(t % 32)) & 1 == 0)) continue;
        const v: f32 = @bitCast(@as(u32, b) << 16);
        if (std.math.isNan(v) or v == -std.math.inf(f32)) continue;
        if (best == null or v > value) {
            best = @intCast(t);
            value = v;
        }
    }
    return best;
}

test "a masked greedy pick: the largest allowed value, the lowest id among equals, none when nothing is allowed" {
    const bf = struct {
        fn of(x: f32) u16 {
            return @intCast(@as(u32, @bitCast(x)) >> 16);
        }
    }.of;
    const ninf: u16 = 0xFF80;
    const zeros: [32]u16 = @splat(0); // bf16 0.0
    var row = [_]u16{ bf(1), bf(5), bf(3), bf(5), ninf, bf(-2), 0x7FC0, bf(4) } ++ zeros;
    try std.testing.expectEqual(@as(?u32, 1), bestAllowed(&row, &.{}));
    try std.testing.expectEqual(@as(?u32, 3), bestAllowed(&row, &.{ 0b1000_1100, 0 })); // 2, 3, 7 allowed
    try std.testing.expectEqual(@as(?u32, 7), bestAllowed(&row, &.{ 0b1000_0100 | 0b0101_0000, 0 })); // 2, 4 (-inf), 6 (NaN), 7
    try std.testing.expectEqual(@as(?u32, 5), bestAllowed(&row, &.{ 0b0011_0000, 0 })); // -inf and -2: -2
    try std.testing.expectEqual(@as(?u32, 33), bestAllowed(&row, &.{ 0, 0b10 })); // a second word
    try std.testing.expectEqual(@as(?u32, null), bestAllowed(&row, &.{ 0b0101_0000, 0 })); // only -inf and NaN
    row[1] = bf(4);
    try std.testing.expectEqual(@as(?u32, 3), bestAllowed(&row, &.{}));
}

test "rules as nemotron/cuda_sampler fills them; temperature 0 is greedy" {
    const s: Sampling = .{ .seed = 1234, .temperature = 0.5, .top_k = 20, .top_p = 0.95 };
    const r = rule(check(s).?);
    try std.testing.expectEqual(@as(f32, 2.0), r.inv_t);
    try std.testing.expectEqual(@as(u32, 20), r.top_k);
    try std.testing.expect(std.math.isInf(r.min_log));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(Rule));
    try std.testing.expectEqual(@as(?Sampling, null), check(.{ .seed = 0, .temperature = 0 }));
}
