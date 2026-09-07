$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Get-Content (Join-Path $root "MODEL_MANIFEST.sha256") | ForEach-Object {
    $parts = $_ -split "\\s+", 2
    $actual = (Get-FileHash (Join-Path $root $parts[1]) -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $parts[0].ToLower()) { throw "Checksum mismatch: $($parts[1])" }
    Write-Host "OK $($parts[1])"
}
