//! (Fox T, diagnostics) Which programs make NaNs and infinities
//! (`VICTORY_NAN_METER=1`). The translator makes every float a program
//! stores through a typed view or exports to a float colour target count
//! itself, before any guard, when it is a NaN or an infinity
//! (`victory_shader::emit::SLOT_NAN`): an atomic add to the program's four
//! dwords in a buffer of ours the CPU maps. Each title pipeline gets its
//! place on first use; the frame lines read and clear the counts.
//! What the programs compute is unchanged.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use victory_shader::emit::NAN_SLOTS;
use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// `VICTORY_NAN_METER=1` (the translator reads the same switch).
pub(crate) fn on() -> bool {
    victory_shader::emit::nan_meter()
}

struct Counts {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: a free-threaded object and its mapping.
unsafe impl Send for Counts {}
// SAFETY: as above.
unsafe impl Sync for Counts {}

fn counts() -> Option<&'static Counts> {
    static MADE: OnceLock<Option<Counts>> = OnceLock::new();
    MADE.get_or_init(|| {
        let host = crate::host::host()?;
        let (buffer, data) = host.shared_buffer(u64::from(NAN_SLOTS) * 16)?;
        // SAFETY: a fresh mapping of that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, NAN_SLOTS as usize * 16) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::info!("NaN meter on (VICTORY_NAN_METER=1): NaNs and infinities counted per program where they are stored");
        Some(Counts { _buffer: buffer, data: data as usize, gpu })
    })
    .as_ref()
}

/// Where the shaders count (its GPU address), for the root UAV.
pub(crate) fn address() -> Option<u64> {
    counts().map(|c| c.gpu)
}

/// Places by title pipeline, and their names.
static PLACES: Mutex<Option<(HashMap<usize, u32>, Vec<Arc<str>>)>> = Mutex::new(None);

/// The place of the title pipeline `pipeline` (its programs' names `label`).
pub(crate) fn slot(pipeline: usize, label: impl FnOnce() -> Arc<str>) -> u32 {
    let mut places = PLACES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, names) = places.get_or_insert_with(|| (HashMap::new(), Vec::new()));
    if let Some(&at) = by.get(&pipeline) {
        return at;
    }
    let at = (names.len() as u32).min(NAN_SLOTS - 1);
    if names.len() < NAN_SLOTS as usize - 1 {
        names.push(label());
    }
    by.insert(pipeline, at);
    at
}

/// For the frame lines: the programs with counts since the last one.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let Some(c) = counts() else { return };
    let names: Vec<Arc<str>> = PLACES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map(|p| p.1.clone()).unwrap_or_default();
    let mut rows: Vec<(u64, String)> = Vec::new();
    let mut all = [0u64; 4];
    for (i, name) in names.iter().enumerate() {
        // SAFETY: inside the mapping; the card may add while this reads and
        // clears (a count lost to the race goes to the next line or nowhere).
        let v: [u32; 4] = unsafe {
            let at = (c.data as *mut u32).add(i * 4);
            let v = [at.read_volatile(), at.add(1).read_volatile(), at.add(2).read_volatile(), at.add(3).read_volatile()];
            if v != [0; 4] {
                std::ptr::write_bytes(at, 0, 4);
            }
            v
        };
        if v == [0; 4] {
            continue;
        }
        for k in 0..4 {
            all[k] += u64::from(v[k]);
        }
        let total = v.iter().map(|&x| u64::from(x)).sum::<u64>();
        rows.push((
            total,
            format!(
                "{} (stores NaN {:.0} inf {:.0}, colour exports NaN {:.0} inf {:.0})",
                name.chars().take(140).collect::<String>(),
                f64::from(v[0]) / frames,
                f64::from(v[1]) / frames,
                f64::from(v[2]) / frames,
                f64::from(v[3]) / frames
            ),
        ));
    }
    if rows.is_empty() {
        return;
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0));
    let f = frames.max(1.0);
    tracing::info!(
        "NaNs and infinities made per frame (VICTORY_NAN_METER): typed stores NaN {:.0} inf {:.0}, colour exports NaN {:.0} inf {:.0}, by {} programs; the most: {}",
        all[0] as f64 / f,
        all[1] as f64 / f,
        all[2] as f64 / f,
        all[3] as f64 / f,
        rows.len(),
        rows.iter().take(12).map(|r| r.1.as_str()).collect::<Vec<_>>().join("; ")
    );
}
