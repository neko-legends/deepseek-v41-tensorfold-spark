//! csf_32x2_v1: ModelOpt NVFP4 block scales in FlashInfer B12X's lossless compressed form (GLM-5.3-Flash-FP4-Local's
//! routed experts), decoded on the host to the plain e4m3 [n, k/16] plane nvfp4.Linear carries.
//!
//! The format is B12X's (local-inference-lab/flashinfer, flashinfer/experimental/b12x/loader/_csf_blocks.py,
//! CsfBlockPlane and logical_csf_components); this is a port of its decode. Per module, five tensors beside the codes:
//!   weight_scale.csf32_bases       u8  [rows]           a row's palette base
//!   weight_scale.csf32_codes       u8  [rows cols / 2]  4-bit offsets from the base, two to a byte (low nibble first)
//!   weight_scale.csf32_page_bases  u32 [pages]          byte offset of a page's first exception record
//!   weight_scale.csf32_offsets     u16 [blocks + pages] per page: each block's record prefix, then the page's total
//!   weight_scale.csf32_exceptions  u16 [records]        low byte: position 2 row + col in the block; high byte: the scale
//! A block is 32 rows by 2 columns (one column at an odd tail). Blocks are visited by 128-row by 4-column tiles (4 x 2
//! blocks, clipped), column pair outer, row block inner; a page holds 32 tiles. Scale = base + offset, unless a record
//! of the block replaces it.

const std = @import("std");

pub const block_rows = 32;
pub const block_cols = 2;
pub const page_tiles = 32;

/// One module's compressed scales as the checkpoint stores them (byte views; offsets/exceptions little-endian u16).
pub const Plane = struct {
    rows: usize,
    cols: usize, // k / 16
    bases: []const u8,
    codes: []const u8,
    page_bases: []const u8,
    offsets: []const u8,
    exceptions: []const u8,
};

pub const Geometry = struct {
    row_blocks: usize,
    col_blocks: usize,
    tiles: usize,
    pages: usize,

    pub fn of(rows: usize, cols: usize) !Geometry {
        if (rows == 0 or rows % block_rows != 0 or cols == 0) return error.BadCsf;
        const rb = rows / block_rows;
        const cb = (cols + 1) / 2;
        const tiles = ((rb + 3) / 4) * ((cb + 1) / 2);
        return .{ .row_blocks = rb, .col_blocks = cb, .tiles = tiles, .pages = (tiles + page_tiles - 1) / page_tiles };
    }

    pub fn blocks(g: Geometry) usize {
        return g.row_blocks * g.col_blocks;
    }
};

/// The sizes `p`'s tensors must have (exceptions: any even byte count, checked against the offsets in decode).
pub fn check(p: Plane) !Geometry {
    const g = try Geometry.of(p.rows, p.cols);
    if (p.bases.len != p.rows or p.codes.len * 2 != p.rows * p.cols or p.page_bases.len != 4 * g.pages or
        p.offsets.len != 2 * (g.blocks() + g.pages) or p.exceptions.len % 2 != 0) return error.BadCsf;
    return g;
}

fn u16At(b: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, b[2 * i ..][0..2], .little);
}

fn u32At(b: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, b[4 * i ..][0..4], .little);
}

/// `p` decoded into `out` (rows * cols bytes, row-major e4m3): bit-exact logical_csf_components + bases + exceptions.
pub fn decode(p: Plane, out: []u8) !void {
    const g = try check(p);
    if (out.len != p.rows * p.cols) return error.BadCsf;
    const records = p.exceptions.len / 2;
    var field: usize = 0; // the next 4-bit offset in the code stream
    var seq: usize = 0; // blocks visited so far
    var rec: usize = 0; // exception records consumed so far
    var tile: usize = 0;
    var r4: usize = 0;
    while (r4 < g.row_blocks) : (r4 += 4) {
        var c2: usize = 0;
        while (c2 < g.col_blocks) : (c2 += 2) {
            const page = tile / page_tiles;
            if (tile % page_tiles == 0) {
                // a page starts its prefixes at 0 and its records where page_bases says: one consecutive arena
                if (u16At(p.offsets, seq + page) != 0 or u32At(p.page_bases, page) != 2 * rec) return error.BadCsf;
            }
            for (c2..@min(c2 + 2, g.col_blocks)) |bc| for (r4..@min(r4 + 4, g.row_blocks)) |br| {
                const width: usize = @min(block_cols, p.cols - 2 * bc);
                for (0..block_rows) |rr| for (0..width) |cc| {
                    const r = br * block_rows + rr;
                    const code = (p.codes[field / 2] >> @intCast(4 * (field % 2))) & 0xf;
                    field += 1;
                    const v = @as(u16, p.bases[r]) + code;
                    if (v > 255) return error.BadCsf;
                    out[r * p.cols + 2 * bc + cc] = @intCast(v);
                };
                const at = seq + page;
                const lo = u16At(p.offsets, at);
                const hi = u16At(p.offsets, at + 1);
                if (hi < lo or hi - lo > block_rows * width) return error.BadCsf;
                var last: i32 = -1;
                for (0..hi - lo) |_| {
                    if (rec >= records) return error.BadCsf;
                    const e = u16At(p.exceptions, rec);
                    rec += 1;
                    const local: usize = e & 0xff;
                    if (local >= 2 * block_rows or local % 2 >= width or @as(i32, @intCast(local)) <= last) return error.BadCsf;
                    last = @intCast(local);
                    out[(br * block_rows + local / 2) * p.cols + 2 * bc + local % 2] = @intCast(e >> 8);
                }
                seq += 1;
            };
            tile += 1;
        }
    }
    if (rec != records or field != p.rows * p.cols) return error.BadCsf;
}

/// A fixture of csf_fixtures.py: u32 rows, cols, records; bases; codes; page_bases; offsets; exceptions; logical.
fn fixture(blob: []const u8) !struct { plane: Plane, logical: []const u8 } {
    const rows = std.mem.readInt(u32, blob[0..4], .little);
    const cols = std.mem.readInt(u32, blob[4..8], .little);
    const records = std.mem.readInt(u32, blob[8..12], .little);
    const g = try Geometry.of(rows, cols);
    var at: usize = 12;
    const take = struct {
        fn f(b: []const u8, a: *usize, n: usize) []const u8 {
            defer a.* += n;
            return b[a.*..][0..n];
        }
    }.f;
    const plane: Plane = .{
        .rows = rows,
        .cols = cols,
        .bases = take(blob, &at, rows),
        .codes = take(blob, &at, rows * cols / 2),
        .page_bases = take(blob, &at, 4 * g.pages),
        .offsets = take(blob, &at, 2 * (g.blocks() + g.pages)),
        .exceptions = take(blob, &at, 2 * records),
    };
    const logical = take(blob, &at, rows * cols);
    if (at != blob.len) return error.BadFixture;
    return .{ .plane = plane, .logical = logical };
}

test "csf_32x2_v1 planes made by B12X's encoder decode to its logical scales" {
    const names = [_][]const u8{ "32x1", "32x2", "64x5", "160x9", "256x8", "128x4", "288x70" };
    inline for (names) |n| {
        const f = try fixture(@embedFile("fixtures/csf32x2/csf_" ++ n ++ ".bin"));
        const out = try std.testing.allocator.alloc(u8, f.logical.len);
        defer std.testing.allocator.free(out);
        try decode(f.plane, out);
        try std.testing.expectEqualSlices(u8, f.logical, out);
    }
}

test "csf geometry: the FP4-Local experts' planes" {
    // gate/up [2048, 2048/16 * 2 = 256], down [4096, 128]: 8192 blocks, 32 pages, 8224 offsets, 262144 code bytes
    for ([_][2]usize{ .{ 2048, 256 }, .{ 4096, 128 } }) |s| {
        const g = try Geometry.of(s[0], s[1]);
        try std.testing.expectEqual(@as(usize, 8192), g.blocks());
        try std.testing.expectEqual(@as(usize, 32), g.pages);
        try std.testing.expectEqual(@as(usize, 262144), s[0] * s[1] / 2);
    }
    try std.testing.expectError(error.BadCsf, Geometry.of(48, 4));
}

test "corrupt planes are refused" {
    const f = try fixture(@embedFile("fixtures/csf32x2/csf_160x9.bin"));
    const out = try std.testing.allocator.alloc(u8, f.logical.len);
    defer std.testing.allocator.free(out);
    var short = f.plane;
    short.exceptions = short.exceptions[0 .. short.exceptions.len - 2];
    try std.testing.expectError(error.BadCsf, decode(short, out));
    var bad_codes = f.plane;
    bad_codes.codes = bad_codes.codes[1..];
    try std.testing.expectError(error.BadCsf, decode(bad_codes, out));
    const pb = try std.testing.allocator.dupe(u8, f.plane.page_bases);
    defer std.testing.allocator.free(pb);
    pb[0] = 2;
    var bad_page = f.plane;
    bad_page.page_bases = pb;
    try std.testing.expectError(error.BadCsf, decode(bad_page, out));
}
