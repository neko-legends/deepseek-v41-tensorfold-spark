//! GLM-5.3-Flash NVFP4 native CUDA at TP over processes (TF_GLM_RANK, NCCL_COMM_ID): rank 0 serves, others follow.
//! ours: also GLM-5.3-Flash-FP4-Local (ModelOpt MIXED_PRECISION, cuda_weights.zig).

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const vision = @import("cuda_vision.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{ "modelopt-nvfp4", "modelopt-mixed" };
// cc-defer: a fixed window; the Python family sizes it from free memory (capacity.admit), port that when serving needs
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fw.prefill_rows;
/// Prompt kernels change arithmetic with a chunk's rows: states keep and resume only on chunk starts.
pub const snap_grid: u32 = prompt_rows;
/// `explain` words this family's open refusals too (where to set the rank and the id).
pub const explains_open = true;

/// The server's reading of this rank's room after its load (the budget streams are admitted by); null if unreadable.
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };
/// `slide` stays false: the server refuses --slide for a family without a learner (no `slide` here).
pub const Options = struct {
    context: usize,
    drafts: bool,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    vision: ?vision.Limits = null, // ours: --vision (the image tower on rank 0, image rows on every rank)
};

/// Ours: rank 0's room when its image tower failed to load (a room no rank reads otherwise): the others stop too.
const tower_failed_room: u64 = std.math.maxInt(u64) - 0x7f;

/// Ours: --vision's default image tokens a request holds: the processor's max_image_tokens, so one image's grid is HF's.
pub const default_image_tokens: u32 = 8000;
/// The smallest room among the other ranks and whose it is.
pub const Peer = struct { bytes: u64, rank: usize };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
    peer_room: ?Peer = null, // rank 0: the server admits streams and kept states by this when it is smaller
    vision: ?*vision.Vision = null, // ours: rank 0's tower with --vision (native/cuda_vision_offer.zig offers it)
};

/// This rank's place: TF_GLM_RANK of TF_GLM_WORLD (both, or neither for a refusal that names them).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_GLM_RANK") orelse return error.NoRank;
    const w = std.c.getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w), 10) catch return error.NoRank };
    return checked(p);
}

/// A place the family serves: a rank inside its world, and at most `ranks.max_world` ranks.
fn checked(p: Place) !Place {
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    if (p.world > ranks.max_world) return error.TooManyRanks;
    return p;
}

/// Bytes this rank's weights take at most, for the server's admission: the checkpoint times its largest axis share.
pub fn weightShare(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64) u64 {
    const p = place() catch return total;
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return total;
    const text = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 24)) catch return total;
    const c = config.parse(a, text) catch return total;
    const f = (shard.Share.of(&c, p.rank, p.world) catch return total).part(&c);
    return @intCast(@as(u128, total) * f[0] / f[1]);
}

/// Slots for `context` plus a verify window, a multiple of 64 so the sparse pool keeps the AOT set.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + fw.window_rows, 64);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
    images: ?*vision.Images = null,
    vision: ?*vision.Vision = null,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const p = try place();
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    const set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.lib = try cuda.nccl.Library.open();
    errdefer own.lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    try own.lib.check(own.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // NCCL_COMM_ID: the same id on every rank
    own.comm = null;
    try own.lib.check(own.lib.api.ncclCommInitRank(&own.comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    errdefer _ = own.lib.api.ncclCommDestroy(own.comm);
    const cap = capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.lib, own.comm, p.rank, p.world, .{ .capacity = cap, .mtp = o.drafts, .settings = set, .vision_tokens = if (o.vision) |v| v.image_tokens else 0 });
    errdefer own.e.deinit();
    own.link = try follow.Link.init(gpa, ctx.d, &own.lib, own.comm, own.e.stream, cap);
    errdefer own.link.deinit();
    // ours: --vision: the rows buffer on every rank, the tower on rank 0 (before the rooms are read: its memory counts)
    own.images = null;
    own.vision = null;
    errdefer if (own.images) |im| im.deinit();
    errdefer if (own.vision) |v| v.deinit();
    var tower_failed: ?anyerror = null; // rank 0's, told the other ranks in the rooms' exchange: all stop together
    if (o.vision) |lim| {
        own.images = try vision.Images.init(gpa, ctx.d, lim.image_tokens, own.e.c.hidden);
        own.e.images = own.images;
        if (p.rank == 0) own.vision = vision.Vision.init(gpa, io, ctx.d, dir, lim) catch |err| blk: {
            std.log.err("--vision: the image tower does not load ({s})", .{@errorName(err)});
            tower_failed = err;
            break :blk null;
        };
    }
    // Rank 0 admits by the smallest room of the ranks; its sizes are the largest (the first ranks take the remainder).
    const rooms = try gpa.alloc(u64, p.world);
    defer gpa.free(rooms);
    const mine: u64 = if (tower_failed != null) tower_failed_room else if (o.room) |r| r.read(r.ctx) orelse 0 else std.math.maxInt(u64);
    try ranks.gather(u64, ctx.d, &own.lib, own.comm, own.e.stream, p.rank, mine, rooms);
    if (tower_failed) |err| return err;
    if (rooms[0] == tower_failed_room) return error.VisionTowerFailed; // rank 0's tower did not load
    var peer: ?Peer = null;
    for (rooms[1..], 1..) |r, q| if (peer == null or r < peer.?.bytes) {
        peer = .{ .bytes = r, .rank = q };
    };
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, then leaves the process
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            std.process.exit(1);
        };
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        std.process.exit(0);
    }
    // Rank 0 alone reads these: TF_GLM_PROMPT_STEPS=0 passes prompts whole, TF_GLM_SHARED=0 runs windows alone.
    const off = struct {
        fn get(name: [*:0]const u8) bool {
            return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        }
    }.get;
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link }, .whole = off("TF_GLM_PROMPT_STEPS"), .alone = off("TF_GLM_SHARED"), .vision = own.vision };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = glm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes(), // each stream past the first makes its own sequence on every rank
        .free_streams = 1, // the first is the sequence the engine made
        .ctx = own,
        .deinit = close,
        .target = own.e,
        .peer_room = peer,
        .vision = own.vision,
    };
}

// Each rank keeps a copy under rank 0's id, in the room rank 0 admitted by (the smallest over the ranks).

/// Bytes a state takes at any position: the whole state arena on each rank.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    _ = at;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes();
}

/// The prompt pass stands at `at` (a mark at a chunk start): copy the state there under a new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at };
    return k;
}

/// A restore outside a prompt pass (the store's `begin`) would leave the other ranks behind, so it refuses.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    own.link.deinit();
    if (own.vision) |v| v.deinit();
    own.e.images = null;
    if (own.images) |im| im.deinit();
    own.e.deinit();
    _ = own.lib.api.ncclCommDestroy(own.comm);
    own.lib.close();
    own.gpa.destroy(own);
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "GLM-5.3-Flash runs at TP over hosts: set TF_GLM_RANK and TF_GLM_WORLD on every host (WORLD 1: one GPU, with TF_GLM_LAYERS for a thin model)",
        error.NoCommId => "GLM-5.3-Flash's ranks meet over NCCL: with TF_GLM_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        error.TooManyRanks => std.fmt.comptimePrint(
            "GLM-5.3-Flash on CUDA runs on at most {d} ranks: set TF_GLM_WORLD to {d} or fewer",
            .{ ranks.max_world, ranks.max_world },
        ),
        error.NotModelOpt, error.UnsupportedQuant => "GLM-5.3-Flash on CUDA reads ModelOpt NVFP4 checkpoints (the log names what config.json has)",
        else => null,
    };
}

test "a place inside its world of at most 16 ranks serves; a larger world is refused by name" {
    try std.testing.expectEqual(Place{ .rank = 15, .world = 16 }, try checked(.{ .rank = 15, .world = 16 }));
    try std.testing.expectError(error.TooManyRanks, checked(.{ .rank = 0, .world = 17 }));
    try std.testing.expectError(error.NoRank, checked(.{ .rank = 2, .world = 2 }));
    try std.testing.expect(explain(null, error.TooManyRanks) != null);
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expectEqual(@as(usize, 330560), capacity(330529));
    try std.testing.expect(capacity(330529) % 64 == 0);
}
