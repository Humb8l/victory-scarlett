# Starts the title with this branch's current settings and memory hooks (the set the README's figures are
# measured with): Series X mode, or Series S with DLSS (-OptiScaler and -Dlss).
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
# -RenderScale the title's internal render scale, 0.2-1.0 of the output (VICTORY_RENDER_SCALE; the title's own
#              is 2/3: 2560x1440 of 3840x2160). 0.5 = 1920x1080 in Series X mode: fewer pixels for the card and
#              about 1 GB less in render targets. With -Dlss only 0.5, 0.4, 1/3 (0.3333) and 0.75 reach DLSS.
# -OptiScaler  the folder of an extracted OptiScaler release, and
# -Dlss        NVIDIA's nvngx_dlss.dll: DLSS in place of the title's upscaler (cool_fox's route,
#              VICTORY_FSR_DLL). The title calls that upscaler only as a Series S, so this starts in Series S mode
#              (720p upscaled to 1080p unless -RenderScale). OptiScaler is set up in target\<Build>\opti.
# -DlssPreset  DLSS render preset with -Dlss (default 11 = K, the transformer model; 0 = DLSS's own choice)
# -More        other settings, last word: -More @{ VICTORY_LOG = 'warn' }
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
    [ValidateRange(0.2, 1.0)][double]$RenderScale = 0,
    [string]$OptiScaler = '',
    [string]$Dlss = '',
    [ValidateRange(0, 15)][int]$DlssPreset = 11,
    [hashtable]$More = @{}
)
$ErrorActionPreference = 'Stop'
if ([bool]$OptiScaler -ne [bool]$Dlss) { throw 'DLSS needs both -OptiScaler <folder> and -Dlss <nvngx_dlss.dll>.' }

function Set-IniValue([string]$Text, [string]$Section, [string]$Key, [string]$Value) {
    $lines = $Text -split "`r?`n"
    $in = $false; $done = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '^\[(.+)\]\s*$') { $in = ($Matches[1] -eq $Section); continue }
        if ($in -and -not $done -and $lines[$i] -match ('^\s*' + [regex]::Escape($Key) + '\s*=')) { $lines[$i] = "$Key=$Value"; $done = $true }
    }
    if (-not $done) { Write-Warning "OptiScaler.ini: no $Key in [$Section]" }
    $lines -join "`r`n"
}

# A copy of victory.exe in a folder of its own with OptiScaler as its dxgi.dll and the DLSS DLL beside it (cool_fox's
# Play-DLSS.ps1). OptiScaler takes the PC FidelityFX calls the title's upscaler is sent to and runs DLSS instead.
function Initialize-Opti([string]$Exe, [string]$OptiDir, [string]$DlssDll, [int]$Preset) {
    $to = Join-Path (Split-Path $Exe -Parent) 'opti'
    New-Item -ItemType Directory -Force $to | Out-Null
    Get-ChildItem $OptiDir -File | Where-Object { $_.Extension -in '.dll', '.ini' -and $_.Name -ne 'OptiScaler.ini' } | Copy-Item -Destination $to -Force
    Get-ChildItem $OptiDir -Directory | Where-Object { $_.Name -ne 'Licenses' } | Copy-Item -Destination $to -Recurse -Force
    Copy-Item (Join-Path $OptiDir 'OptiScaler.dll') (Join-Path $to 'dxgi.dll') -Force
    Remove-Item (Join-Path $to 'OptiScaler.dll') -Force -ErrorAction SilentlyContinue
    Copy-Item $DlssDll (Join-Path $to 'nvngx_dlss.dll') -Force
    $iniPath = Join-Path $to 'OptiScaler.ini'
    if (-not (Test-Path -LiteralPath $iniPath -PathType Leaf)) { Copy-Item -LiteralPath (Join-Path $OptiDir 'OptiScaler.ini') -Destination $iniPath }
    $ini = Get-Content -LiteralPath $iniPath -Raw
    $ini = Set-IniValue $ini 'Upscalers' 'Dx12Upscaler' 'dlss'
    $ini = Set-IniValue $ini 'FrameGen' 'Enabled' 'false'
    $ini = Set-IniValue $ini 'FrameGen' 'FGInput' 'nofg'
    $ini = Set-IniValue $ini 'FrameGen' 'FGOutput' 'nofg'
    $ini = Set-IniValue $ini 'DLSS' 'RenderPresetOverride' ($(if ($Preset -gt 0) { 'true' } else { 'false' }))
    $ini = Set-IniValue $ini 'DLSS' 'RenderPresetForAll' ([string]$Preset)
    Set-Content -LiteralPath $iniPath -Value $ini -NoNewline
    Copy-Item $Exe (Join-Path $to 'victory.exe') -Force
    $to
}
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
if ($RenderScale -gt 0) { $set.VICTORY_RENDER_SCALE = $RenderScale.ToString([Globalization.CultureInfo]::InvariantCulture) }
$console = @()
$mode = 'Series X'
if ($Dlss) {
    $optiDir = (Resolve-Path -LiteralPath $OptiScaler).Path
    if (-not (Test-Path -LiteralPath (Join-Path $optiDir 'OptiScaler.dll'))) { throw "No OptiScaler.dll in $optiDir" }
    $opti = Initialize-Opti $Victory $optiDir (Resolve-Path -LiteralPath $Dlss).Path $DlssPreset
    $Victory = Join-Path $opti 'victory.exe'
    $set.VICTORY_FSR_DLL = Join-Path $opti 'amd_fidelityfx_dx12.dll'
    $console = @('--console', 'series-s')
    $mode = "Series S with DLSS (preset $DlssPreset)"
}
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
Write-Host ("{0} mode, {1} build, ray tracing {2}, memory hooks {3}, render scale {4}" -f $mode, $Build,
    $(if ($RayTracing) { 'on' } else { 'off' }), $(if ($NoHooks) { 'off' } else { 'on' }),
    $(if ($RenderScale -gt 0) { $RenderScale } else { "the title's" }))
Write-Host "Cache: $Cache"
Write-Host "Saves: $Data"
& $Victory run $exe --execute @console
exit $LASTEXITCODE
