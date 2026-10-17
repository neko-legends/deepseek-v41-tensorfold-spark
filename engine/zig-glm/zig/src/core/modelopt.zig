//! ModelOpt NVFP4 config check (groups of 16); tensors say which linears are NVFP4, not ignore (423acf37's BF16 MTP).
//! MIXED_PRECISION (nvidia/GLM-5.3-Flash-FP4-Local) is admitted when every quantized_layers entry is a format the
//! loader reads: NVFP4 in groups of 16 (plain scales or csf_32x2_v1), W4A16_NVFP4, MXFP8 (1 x 32) and FP8_PB_WO (128 x 128).

const std = @import("std");

/// The NVFP4 group size ModelOpt writes for this format.
pub const nvfp4_group = 16;

/// Why the check refused the checkpoint, kept for the caller's log line (tests read it instead).
pub const Why = struct {
    buf: [320]u8 = undefined,
    len: usize = 0,

    /// Keep the reason, cut short if it is long.
    pub fn set(self: *Why, comptime fmt: []const u8, args: anytype) void {
        var w: std.Io.Writer = .fixed(&self.buf);
        w.print(fmt, args) catch {};
        self.len = w.end;
    }

    pub fn text(self: *const Why) []const u8 {
        return self.buf[0..self.len];
    }
};

/// config.json's quantization_config (top level, else text_config's) checked as ModelOpt NVFP4.
pub fn admit(gpa: std.mem.Allocator, config_json: []const u8, why: *Why) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{});
    const q = block(root) orelse {
        why.set("config.json has no ModelOpt quantization_config", .{});
        return error.NotModelOpt;
    };
    const algo = text(q.get("quant_algo")) orelse "";
    if (std.ascii.eqlIgnoreCase(algo, "MIXED_PRECISION")) return admitMixed(q, why);
    if (!std.ascii.eqlIgnoreCase(algo, "NVFP4")) {
        const named = if (algo.len > 0) algo else "without quant_algo";
        why.set("ModelOpt {s}: this engine reads ModelOpt NVFP4 (nvidia/GLM-5.3-Flash-NVFP4's layout)", .{named});
        return error.UnsupportedQuant;
    }
    const groups = object(q.get("config_groups")) orelse {
        why.set("ModelOpt NVFP4 without config_groups", .{});
        return error.UnsupportedQuant;
    };
    var it = groups.iterator();
    while (it.next()) |g| {
        const w = if (object(g.value_ptr.*)) |o| object(o.get("weights")) else null;
        const bits = if (w) |o| int(o.get("num_bits")) else null;
        const size = if (w) |o| int(o.get("group_size")) else null;
        const kind = if (w) |o| text(o.get("type")) orelse "" else "";
        if (bits != 4 or size != nvfp4_group or !std.mem.eql(u8, kind, "float")) {
            why.set(
                "ModelOpt NVFP4 group {s}: weights {?d}-bit {s} in groups of {?d}, " ++
                    "expected 4-bit float in groups of 16",
                .{ g.key_ptr.*, bits, kind, size },
            );
            return error.UnsupportedQuant;
        }
    }
}

/// ours: nvidia/GLM-5.3-Flash-FP4-Local's config.json, for the tests of the families that load it.
pub const fp4_local_config = @embedFile("fixtures/glm53_flash_fp4_local/config.json");

/// The per-module formats of a MIXED_PRECISION checkpoint (ours: the FP4-Local port's loader reads each).
pub const Mixed = enum { nvfp4, nvfp4_csf, w4a16_nvfp4, mxfp8, fp8_pb };

/// One quantized_layers entry's format, else null (the caller names the entry).
pub fn mixedFormat(entry: std.json.ObjectMap) ?Mixed {
    const algo = text(entry.get("quant_algo")) orelse return null;
    const group = int(entry.get("group_size"));
    const blocks = if (entry.get("weight_block_size")) |v| (if (v == .array) v.array.items else null) else null;
    const blk: [2]?i64 = if (blocks) |b| (if (b.len == 2) .{ int(b[0]), int(b[1]) } else .{ -1, -1 }) else .{ null, null };
    if (std.mem.eql(u8, algo, "NVFP4") or std.mem.eql(u8, algo, "W4A16_NVFP4")) {
        if (group != nvfp4_group or blocks != null) return null;
        const enc = text(entry.get("weight_scale_encoding"));
        if (std.mem.eql(u8, algo, "W4A16_NVFP4")) return if (enc == null) .w4a16_nvfp4 else null;
        if (enc == null) return .nvfp4;
        return if (std.mem.eql(u8, enc.?, "csf_32x2_v1")) .nvfp4_csf else null;
    }
    if (std.mem.eql(u8, algo, "MXFP8")) {
        // either spelling ModelOpt writes: group_size 32, or weight_block_size [1, 32]
        const ok = if (blocks != null) (blk[0] == 1 and blk[1] == 32 and (group == null or group == 32)) else group == 32;
        return if (ok) .mxfp8 else null;
    }
    if (std.mem.eql(u8, algo, "FP8_PB_WO")) return if (blk[0] == 128 and blk[1] == 128) .fp8_pb else null;
    return null;
}

fn admitMixed(q: std.json.ObjectMap, why: *Why) !void {
    const layers = object(q.get("quantized_layers")) orelse {
        why.set("ModelOpt MIXED_PRECISION without quantized_layers", .{});
        return error.UnsupportedQuant;
    };
    var it = layers.iterator();
    while (it.next()) |e| {
        const entry = object(e.value_ptr.*);
        if (entry == null or mixedFormat(entry.?) == null) {
            const algo = if (entry) |o| text(o.get("quant_algo")) orelse "?" else "?";
            why.set("ModelOpt MIXED_PRECISION {s}: {s} is not a format this engine reads " ++
                "(NVFP4 / W4A16_NVFP4 in groups of 16, csf_32x2_v1 scales, MXFP8 1x32, FP8_PB_WO 128x128)", .{ e.key_ptr.*, algo });
            return error.UnsupportedQuant;
        }
    }
}

fn block(root: std.json.Value) ?std.json.ObjectMap {
    const top = object(root) orelse return null;
    for ([_]?std.json.ObjectMap{ top, object(top.get("text_config")) }) |source| {
        const q = object((source orelse continue).get("quantization_config")) orelse continue;
        if (std.ascii.eqlIgnoreCase(text(q.get("quant_method")) orelse "", "modelopt")) return q;
    }
    return null;
}

fn object(v: ?std.json.Value) ?std.json.ObjectMap {
    const x = v orelse return null;
    return if (x == .object) x.object else null;
}

fn text(v: ?std.json.Value) ?[]const u8 {
    const x = v orelse return null;
    return if (x == .string) x.string else null;
}

fn int(v: ?std.json.Value) ?i64 {
    const x = v orelse return null;
    return if (x == .integer) x.integer else null;
}

test "nvidia/GLM-5.3-Flash-NVFP4's config is admitted" {
    var why: Why = .{};
    try admit(std.testing.allocator, @embedFile("fixtures/glm53_flash_nvfp4/config.json"), &why);
}

/// admit() must refuse `json` with `err`, its reason naming `word`.
fn expectRefused(json: []const u8, err: anyerror, word: []const u8) !void {
    var why: Why = .{};
    try std.testing.expectError(err, admit(std.testing.allocator, json, &why));
    if (std.mem.indexOf(u8, why.text(), word) == null) {
        std.debug.print("refusal \"{s}\" does not name \"{s}\"\n", .{ why.text(), word });
        return error.TestUnexpectedResult;
    }
}

test "other ModelOpt formats, and configs without ModelOpt, are refused by name" {
    try expectRefused("{\"text_config\": {}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused("{\"quantization_config\": {\"quant_method\": \"exl3\"}}", error.NotModelOpt, "no ModelOpt");
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"FP8\"}}",
        error.UnsupportedQuant,
        "ModelOpt FP8",
    );
    // MIXED_PRECISION without its per-module map, or with a format the loader does not read, is refused by name
    try expectRefused(
        "{\"quantization_config\": {\"quant_method\": \"modelopt\", \"quant_algo\": \"MIXED_PRECISION\"}}",
        error.UnsupportedQuant,
        "MIXED_PRECISION",
    );
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"a.experts": {"quant_algo": "NVFP4", "group_size": 16, "weight_scale_encoding": "csf"}}}}
    , error.UnsupportedQuant, "a.experts");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.q_proj": {"quant_algo": "INT4_AWQ", "group_size": 128}}}}
    , error.UnsupportedQuant, "INT4_AWQ");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers":
        \\ {"m.o_proj": {"quant_algo": "FP8_PB_WO", "weight_block_size": [64, 64]}}}}
    , error.UnsupportedQuant, "m.o_proj");
    try expectRefused(
        \\{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 32}}}}}
    , error.UnsupportedQuant, "groups of 32");
    var why: Why = .{};
    try admit(std.testing.allocator,
        \\{"text_config": {"quantization_config": {"quant_method": "modelopt", "quant_algo": "nvfp4", "config_groups":
        \\ {"g": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}}}}
    , &why);
}

test "nvidia/GLM-5.3-Flash-FP4-Local's MIXED_PRECISION config is admitted, every entry a known format" {
    const json = fp4_local_config;
    var why: Why = .{};
    try admit(std.testing.allocator, json, &why);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();
    var counts: [5]usize = @splat(0);
    var it = parsed.value.object.get("quantization_config").?.object.get("quantized_layers").?.object.iterator();
    while (it.next()) |e| counts[@intFromEnum(mixedFormat(e.value_ptr.object).?)] += 1;
    // routed experts 42 (csf), MTP experts + vision MLP/merger 77 (W4A16), MXFP8 205, FP8_PB_WO 170 (see GLM-PORT-LOG.md)
    try std.testing.expectEqualSlices(usize, &.{ 0, 42, 77, 205, 170 }, &counts);
}
