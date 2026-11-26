# Starts the title in Series X mode with this branch's current settings and memory hooks
# (the set the README's figures are measured with).
#
#   .\scripts\play-scarlett.ps1 -Game 'D:\Games\MyDump'
#
# -Game        your own dump: the folder with game_scarlett_master.exe (or that file itself).
#              Remembered in game-folder.txt next to the repository's README, so later starts need no -Game.
# -Build       diag (default: the build all measurements are of) or release
# -Victory     another victory.exe than target\<Build>\victory.exe
# -Cache       shader and pipeline caches (default: %LOCALAPPDATA%\victory-scarlett\cache). A first run fills them
#              and is slow; faces can be missing in that first run and are right from the second.
# -Data        saves (default: %LOCALAPPDATA%\victory-scarlett\data)
# -RayTracing  the title's own ray tracing (VICTORY_BVH_BUILDER=1): better light, much slower. Off by default.
# -AutoStart   presses through the start screens by itself, for a fresh start with no save
# -NoHooks     without the memory hooks: the first thing to try if something looks wrong
# -CardSteps   0, 1 or 2 (default 2): which steps that read memory still in flight are done on the card.
#              2 also does the title's buffer copies there; the skinning passes write that memory and
#              earlier reviews saw bodies stretch with it. 1 is the first thing to try if meshes stretch.
# -More       other settings, last word: -More @{ VICTORY_LOG = 'warn' }
#
# Run it from a normal window, not as administrator.
param(
    [string]$Game = '',
    [ValidateSet('diag', 'release')][string]$Build = 'diag',
    [string]$Victory = '',
    [string]$Cache = '',
    [string]$Data = '',
    [switch]$RayTracing,
    [switch]$AutoStart,
    [switch]$NoHooks,
    [ValidateSet('0', '1', '2')][string]$CardSteps = '2',
    [hashtable]$More = @{}
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$remembered = Join-Path $root 'game-folder.txt'

if (-not $Game -and (Test-Path -LiteralPath $remembered)) { $Game = (Get-Content -LiteralPath $remembered -TotalCount 1) }
if (-not $Game) { throw "Say where your dump is: .\scripts\play-scarlett.ps1 -Game 'D:\Games\MyDump'" }
$Game = $Game.Trim().Trim('"')
if ((Test-Path -LiteralPath $Game -PathType Leaf) -and $Game.EndsWith('.exe')) { $Game = Split-Path -Parent $Game }
$exe = Join-Path $Game 'game_scarlett_master.exe'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "No game_scarlett_master.exe in $Game. -Game is the folder of your dump." }
Set-Content -LiteralPath $remembered -Value $Game -Encoding utf8

if (-not $Victory) { $Victory = Join-Path $root "target\$Build\victory.exe" }
if (-not (Test-Path -LiteralPath $Victory -PathType Leaf)) {
    throw "No $Victory. Build it first: cargo build --profile diag -p victory --features diagnostics --locked"
}
if (-not $Cache) { $Cache = Join-Path $env:LOCALAPPDATA 'victory-scarlett\cache' }
if (-not $Data) { $Data = Join-Path $env:LOCALAPPDATA 'victory-scarlett\data' }
New-Item -ItemType Directory -Force -Path $Cache, $Data | Out-Null

# Settings left in this window by an earlier start go: only what is below counts.
Get-ChildItem Env: | Where-Object { $_.Name -like 'VICTORY_*' } | ForEach-Object { Remove-Item "Env:$($_.Name)" }

$set = [ordered]@{
    VICTORY_GAME_DIR  = $Game
    VICTORY_CACHE_DIR = $Cache
    VICTORY_DATA_DIR  = $Data

    # The standing set of this fork's runs (some names come from its older tree and are ignored here)
    VICTORY_JOIN_COPIES = '1'; VICTORY_REFRESH_ONCE = '1'; VICTORY_CARD_REHASH_RUNS = '1'; VICTORY_BACK_OWN_VIEW = '1'
    VICTORY_WIDE_MERGE = '1'; VICTORY_MIRROR_FAST = '1'; VICTORY_FRAME_LOG = '1'; VICTORY_CARD_METER = '1'
    VICTORY_PIPELINE_WAIT_MS = '20'; VICTORY_ASYNC_SUBMIT = '1'; VICTORY_QUEUE2 = '1'; VICTORY_INDIRECT_WINDOW = '1'
    VICTORY_TEXTURE_NO_WAIT = '1'; VICTORY_COPY_PLAIN = '1'; VICTORY_WIDE_TEXTURES = '1'; VICTORY_INDIRECT_GATHER = '1'
    VICTORY_LATE_WAIT_LISTS = '1024'; VICTORY_CARD_FILLS_BATCH = '0'; VICTORY_GATHER_DISPATCH = '0'

    # cool_fox's launcher set, without Series S, VICTORY_THIN and the stand-in ambient light; the keep-busy loop off
    VICTORY_RELAY = '2'; VICTORY_WIDE_BACK = '1'; VICTORY_RELAY_COPY = '1'; VICTORY_CARD_STEPS = '2'
    VICTORY_INDIRECT_BATCH = '2'; VICTORY_MAP_THREADS = '1'; VICTORY_CARD_AWAKE = '0'; VICTORY_SHOW_AFTER = '1'
    VICTORY_JOIN_PARTS = '8'; VICTORY_UPLOAD_AHEAD = '1'; VICTORY_CARD_BUFFERS_MIN_KB = '64'

    # Textures keep only the levels their views show (the base's default; cool_fox's launcher turned it off)
    VICTORY_LEVELS_ON_DEMAND = '0'

    # This fork's memory and upload switches
    VICTORY_POOL_DISCARD = '1'; VICTORY_FILL_TITLE = '1'; VICTORY_PARTIAL_DELTA = '1'; VICTORY_BAKE_DELTA = '1'
    VICTORY_MAP_LOCK_FINE = '1'; VICTORY_POOL_CAP = '1'

    # Draw set-up: table entries done earlier in a list are not done again; an aliased texture's copy follows its
    # source's writes within a list (a stale read in the base)
    VICTORY_TABLE_MEMO = '1'; VICTORY_LIST_ALIAS_EXACT = '1'

    # Every NaN and infinity that image and buffer stores write is made 0 / +-65504 (=1 takes only the
    # NaN an AMD card makes; an NVIDIA card's NaN went through and drew the sun black)
    VICTORY_NAN_GUARD_STORES = '2'
}
if (-not $NoHooks) {
    # The memory hooks
    $set.VICTORY_GPU_ONLY = '1'; $set.VICTORY_KNOWN_WRITES = '1'
}
$set.VICTORY_CARD_STEPS = $CardSteps
if ($RayTracing) { $set.VICTORY_BVH_BUILDER = '1' }
if ($AutoStart) {
    $set.VICTORY_AUTO_INPUT = 'd>=15,rep(A,1,d<10),w3,d>=20,w1,rep(A,1,d>=35),w1.5,Down,w7,Down,w0.6,A,w1.5,A,w0.7,Down,w0.5,A,w0.7,Down,w0.5,A'
}
foreach ($k in $More.Keys) { $set[$k] = [string]$More[$k] }
foreach ($k in $set.Keys) { Set-Item -Path "Env:$k" -Value $set[$k] }

# The layer's remapping slows to about twice its time per call when free RAM runs low (fox-port-C: 1.7 GB free in
# the long frames, 4.1 GB in the others); a 32 GB PC has little to spare beside a ~33 GB commit peak.
$os = Get-CimInstance Win32_OperatingSystem
$freeGB = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
if ($freeGB -lt 16) {
    Write-Warning "Only $freeGB GB of RAM free: long frames get more likely. The largest other programs:"
    Get-Process | Where-Object { $_.Name -ne 'victory' } | Sort-Object PrivateMemorySize64 -Descending | Select-Object -First 5 |
        ForEach-Object { Write-Warning ("  {0}: {1:N1} GB" -f $_.Name, ($_.PrivateMemorySize64 / 1GB)) }
}
Write-Host ("Series X mode, {0} build, ray tracing {1}, memory hooks {2}" -f $Build, $(if ($RayTracing) { 'on' } else { 'off' }), $(if ($NoHooks) { 'off' } else { 'on' }))
Write-Host "Cache: $Cache"
Write-Host "Saves: $Data"
& $Victory run $exe --execute
exit $LASTEXITCODE
