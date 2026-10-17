//! csf_32x2_v1 decode of real checkpoint planes for the bit-exact check against B12X's decoder (tools/glm/csf_check.sh):
//! `csf_decode DIR ROWS COLS MODULE...` reads DIR/MODULE.weight_scale.csf32_{bases,codes,page_bases,offsets,exceptions}.bin
//! (raw tensor bytes) and prints `sha256  MODULE` of each decoded e4m3 [ROWS, COLS] plane.

const std = @import("std");
const csf = @import("csf");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 5) {
        std.debug.print("usage: csf_decode DIR ROWS COLS MODULE...\n", .{});
        std.process.exit(2);
    }
    const rows = try std.fmt.parseInt(usize, args[2], 10);
    const cols = try std.fmt.parseInt(usize, args[3], 10);
    const out = try gpa.alloc(u8, rows * cols);
    defer gpa.free(out);
    for (args[4..]) |module| {
        var parts: [5][]u8 = undefined;
        for (&parts, [_][]const u8{ "bases", "codes", "page_bases", "offsets", "exceptions" }) |*p, what| {
            const path = try std.fmt.allocPrint(arena, "{s}/{s}.weight_scale.csf32_{s}.bin", .{ args[1], module, what });
            p.* = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 26));
        }
        try csf.decode(.{ .rows = rows, .cols = cols, .bases = parts[0], .codes = parts[1], .page_bases = parts[2], .offsets = parts[3], .exceptions = parts[4] }, out);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(out, &digest, .{});
        std.debug.print("{x}  {s}\n", .{ &digest, module });
    }
}
