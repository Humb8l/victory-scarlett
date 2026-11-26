//! D3D12.X on host D3D12.
//!
//! D3D12.X records commands from code inlined into the title, which writes
//! packets into the command list while its put pointer is below a limit and
//! calls the list's method otherwise. Our lists keep the limits at zero, so
//! every command arrives here as a call, to be run on host D3D12. The
//! title's GPU addresses are CPU addresses (console memory is unified).
//!
//! [`device`] is the device object titles create, [`dxgi`] its adapter side,
//! [`resource`] heaps and resources, [`fence`] fences, [`queue`] command
//! queues, [`list`] command allocators and lists, [`descriptor`]
//! descriptor heaps, [`root`] root signatures, [`query`] query heaps,
//! [`signature`] command signatures, [`pipeline`] pipeline states,
//! [`frame`] frame pacing, [`object`] what all console graphics objects
//! share, [`console`] which console model the title sees. Planned:
//! `present` (swap chain).

#[cfg(all(windows, feature = "diagnostics"))]
mod capture;
#[cfg(windows)]
mod cardmeter;
#[cfg(windows)]
mod cpugroups;
pub mod hangwatch;
pub mod console;
#[cfg(all(windows, feature = "diagnostics"))]
mod copystat;
pub mod descriptor;
pub mod device;
#[cfg(windows)]
mod drvwatch;
#[cfg(windows)]
mod draw;
pub mod dxgi;
pub mod fence;
pub mod frame;
pub mod framestream;
#[cfg(all(windows, feature = "diagnostics"))]
mod texdetail;
pub mod gpuview;
pub mod mapfast;
#[cfg(windows)]
pub mod framelog;
#[cfg(windows)]
pub mod faults;
#[cfg(windows)]
mod declstat;
#[cfg(windows)]
mod heap_marks;
#[cfg(windows)]
mod heap_pass;
#[cfg(windows)]
mod host;
#[cfg(windows)]
mod indirect;
#[cfg(windows)]
mod inplace;
pub mod list;
pub mod object;
pub mod pipeline;
#[cfg(windows)]
mod plaincopy;

/// (F19 meter) Frames presented so far (`frame::current`).
pub fn frames_presented() -> u64 {
    frame::current()
}

/// (`VICTORY_MAP_PREFAULT`, for the hle side) One byte of each 4 KiB page of
/// `[at, at + len)` read with no lock, a missing page ending it harmlessly:
/// the pages read, and whether it ended early.
#[cfg(windows)]
pub fn prefault_touch(at: usize, len: usize) -> (usize, bool) {
    plaincopy::touch(at, len)
}
#[cfg(windows)]
mod primitive;
#[cfg(all(windows, feature = "diagnostics"))]
mod profile;
pub mod query;
pub mod queue;
#[cfg(windows)]
pub mod readback;
#[cfg(windows)]
mod ranges;
#[cfg(windows)]
mod queue2;
pub mod resource;
pub mod root;
#[cfg(windows)]
mod census;
#[cfg(windows)]
mod sampler;
#[cfg(windows)]
pub mod shader_cache;
#[cfg(windows)]
mod memory;
#[cfg(windows)]
mod shadow;
#[cfg(windows)]
mod tiles;
#[cfg(windows)]
mod shell;
pub mod signature;
#[cfg(windows)]
mod texture;
#[cfg(windows)]
mod timeline;
#[cfg(windows)]
mod watch;
#[cfg(windows)]
mod views;
#[cfg(windows)]
pub mod chunks;

#[cfg(windows)]
pub use host::on_failure;
#[cfg(windows)]
pub use census::title_memory_reserved;
#[cfg(windows)]
pub use inplace::{title_memory_freeing, title_memory_section, title_memory_view};
#[cfg(windows)]
pub use shell::{use_title_dir, window_destroyed};
#[cfg(windows)]
pub use watch::{
    title_memory_committed, title_memory_reading, title_memory_remapped, title_memory_remapping, title_memory_unmapped,
    title_memory_written, writing as kernel_will_write,
};
