# RAM and VRAM plan for one PC: RTX 5060 Ti 16 GB, Ryzen 7 7800X3D, 32 GB RAM (2026-11-26)

Branch `nvidia-black-sun` of victory-scarlett, Series X mode, `Play-Scarlett.cmd`. Nothing below has been run yet.
Figures marked [run] come from this PC's two runs of 2026-11-26 (`target\diag\last-run.log`); the others from this
fork's own notes (`memory-audit.md`, `vram-plan.md`, `fox-port-C.md`), measured on a 4080 SUPER / 32 GB PC.

## What this PC shows

| | value | source |
|---|---|---|
| Free RAM during play (Mandy save, open world) | **1.4 GB** | [run] framelog "1403 MB free" |
| Other programs at the same time | World of Warcraft 8.2 GB private, Bitdefender 0.9, Discord 0.8, SteelSeries 0.6, Fluxer 0.6, Overwolf 0.3 | [run] process list |
| Remapping per frame | 41.6 ms in 103 calls (of a 94 ms frame) | [run] |
| Card in use / budget | 10.8 / 14.1 GB (Mandy), 12.4 / 14.4 GB (first run) | [run] memory housekeeping |
| Texture copies let go for room / made again | 7.7 GB / 0.47 GB (first run, 7 min) | [run] |
| Card busy per frame | 46 ms; idle 40 ms, of which 28 ms while the title records | [run] |
| Commit limit | 60 GB (32 RAM + page files of 17 + 5 + 5 GB on C:, E:, F:) | system |
| Process private (their PC) | 33 GB mid-run, 36 GB peak, mostly the card's contents counted again | memory-audit §0 |

Two facts decide the order:
1. **RAM is the tighter limit on this PC, not VRAM.** With 1.4 GB free, remapping takes about twice its time per
   call (fox-port-C: 153 us against 78, long frames at 1.7 GB free and the others at 4.1 GB). The card still had
   3.3 GB to spare in the Mandy run.
2. **Every MB the layer holds on the card is also a MB of commit** (memory-audit §0). So VRAM saved is commit saved,
   and fewer pixels help both.

## Step 0: no code, before any run (about 8-10 GB RAM back)

- Close World of Warcraft, and Overwolf, the SteelSeries capture and Discord's overlay if they are not needed. That is
  8-10 GB of RAM: more than all the steps below together. The launcher now warns under 16 GB free.
- Do not run the GDeflate conversion or another emulator at the same time.
- Keep the page file system-managed on the fastest NVMe drive; the commit limit (60 GB) is not the problem today.

Check: the launcher's warning is gone; "MB free" in the framelog stays over 4 GB; the remapping ms per frame falls.

## Step 1: existing switches, one run each (about 1.5-2.5 GB commit, 0.4-0.7 GB card)

From memory-audit §5. None of these changes the shader translation, and none trades picture.

| Run | Switches (`-More @{...}`) | Expected | Counter that must move |
|---|---|---|---|
| 1a | `VICTORY_HEAP_PACK=1` `VICTORY_LIST_POOL_IDLE_S=10` `VICTORY_WINDOW_UNMAP=1` | 0.4-0.7 GB card and commit (holes), 0.2-0.4 GB (idle lists), 0-1 GB (windows) | "copies let go to empty a sparse heap" > 0; "host lists idle" < 100; "windows let go on unmap" > 0 |
| 1b | 1a + `VICTORY_WINDOW_KB=1024` `VICTORY_WINDOW_TAILS=1` | 0.9-1.3 GB of window bytes never asked for | windows alive MB close to "asked" MB. Risk: interface vertices, look at the HUD |

Must not get worse: "made again ... in the last 5 s", frame times, frames over 1.5x the usual.
Not worth trying (rejected or already gone under pressure): `KEEP_FREED=0`, `STAGING_SPARE_BLOCKS=8`,
`VRAM_RAW_PCT`, `UNMAP_EXACT=1`, `TEXTURE_RAM_CAPS=0`.

## Step 2: fewer pixels (the largest lever on this card)

The title renders at 2560x1440 in Series X mode whatever display mode it is given. Render targets take 2.4-3.4 GB of
the card (vram-plan §6), and each is counted again in commit. On this card they also cost GPU time: 46 ms a frame.

- Port `VICTORY_RENDER_SCALE` from cool_fox (commit 5fd0377, `victory-loader/src/hooks.rs`, about 280 lines).
  It pins the title's internal render scale; this fork's notes name the same lead (render size globals at
  0x1499fecf8 / 0x1499fecfc, game-render-memory-model §5).
- Port the DLSS route from cool_fox (154fddf, about 575 lines: `VICTORY_FSR_DLL`, `fidelityfx.rs`) so a lower
  internal resolution is upscaled by DLSS through OptiScaler and not by the title's own pass.
- Expected at 1920x1080 internal (56 % of the pixels): about 1.0-1.5 GB of targets back on the card and in commit,
  and a large part of the 46 ms of card time. To be measured, not assumed.
- Risk: the patch is title-version specific; the hook refuses a build it does not recognise.

## Step 3: code from this fork's plans (RAM first)

| # | Change | Gain | Where | Risk |
|---|---|---|---|---|
| 3a | Staging blocks: a rest time per pool (start at 250 ms), `pack = true` for staging, a "staging blocks by occupancy" meter | 1.0-2.5 GB RAM at bursts, exactly when free RAM falls to 0.6-1.3 GB | `pool.rs:13-18`, `:428`, `:532` | wrong texels if a list is reset before its copy ran |
| 3b | Name the 6.2 GB nobody owns (4 x 1 GiB, 40 x 32 MB, 99 x 32 MB write-combined): one meter line | explains it first; likely send buffers and read-back | `copystat.rs:707` | none (meter) |
| 3c | Send buffers: cap the spare at 128 MB on a 32 GB PC instead of 768 MB | up to 0.6 GB RAM | `host/send.rs:123` | after 3b's meter |
| 3d | One room rule on net use, marks 88/82 to about 95/92, a refusal no longer purges 256 MB; keep 3.5 GB for targets; stop at raw 96 % | 1.2-1.9 GB more for textures, fewer copies made again (7.7 GB let go in 7 min here) | `resource.rs` `marks_pct`, `room`, `fallback_need` | raw past the budget: Windows moves card memory to RAM, frame time collapse. Do after step 2, with 3.3 GB to spare |
| 3e | Move live copies out of the sparsest texture heaps card-to-card, then release the heap | 0.4-2.4 GB card and commit | `resource.rs` `TextureHeap`, `trim_texture_heaps` | a slot not remade shows a freed texture |

## Step 4: GDeflate (Neptune only, for comparison)

The verified pack is ready (`%LOCALAPPDATA%\Neptune-GDeflate\gdeflate-bridge\indexed.gdeflate`, 50.4 GiB, 306,587
streams, backend `rtx_io`). It takes texture decoding off the CPU; it does not lower RAM or VRAM (DirectStorage adds
its own staging in VRAM, 128 MB or more). Measure it in Neptune, on the same save, against
`-GDeflateBackend original`.
Disk: 23 GB free on C:. The sidecars (`sidecars-streamed-nvcomp`, 48.7 GB) can be deleted once Neptune has run with
the pack; they are only needed to add more archives later.

## Order and what each run must report

1. Step 0, then a reference run on the Mandy save (same place, 5 minutes): free RAM, remapping ms, card in use, made
   again, usual frame, long frames.
2. Run 1a, then 1b, each against the reference.
3. Step 2 (port, then a run at 1920x1080 internal).
4. Step 3a-3c, then 3d and 3e.

Honest target: after steps 0-2, free RAM stays over 6 GB in play and the card keeps 1.5 GB of headroom. Steps 1 and 3
alone bring peak commit from about 36 to about 31-32 GB; they do not fix the frame rate by themselves.
