# Every header compiles on its own.
#
# The convention here puts angle-bracket includes first and quoted project
# includes after them, which is readable but gives up what Google's related-header
# -first order buys: a header that fails to include its own dependencies still
# compiles, because the system headers above it happen to supply them. This check
# buys that back. It compiles a translation unit whose entire content is one
# #include, so nothing is in scope that the header did not bring itself.
#
# Torch and raylib headers are skipped - they need include paths this check does
# not carry, and the point is the project's own headers.
#
#   powershell -NoProfile -File docs/check_headers.ps1

$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $env:TEMP "snakeNN_headers"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null
Copy-Item (Join-Path $root "src/*.h") $work

$needs_external = "az_network.h", "network_evaluator.h", "board_render.h", "neural_network.h"
$failures = 0
$checked = 0

foreach ($header in Get-ChildItem $work -Filter *.h | Sort-Object Name)
{
    if ($needs_external -contains $header.Name)
    {
        Write-Host ("  {0,-24} skipped - needs LibTorch or raylib include paths" -f $header.Name)
        continue
    }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($header.Name)
    $unit = Join-Path $work "selftest_$stem.cpp"
    Set-Content $unit -Encoding ASCII -Value "#include `"$($header.Name)`"`n"

    Push-Location $work
    $output = cmd /c "cl /nologo /std:c++20 /EHsc /permissive- /c selftest_$stem.cpp 2>&1"
    $code = $LASTEXITCODE
    Pop-Location
    $checked++
    if ($code -eq 0)
    {
        Write-Host ("  {0,-24} self-contained" -f $header.Name)
    }
    else
    {
        Write-Host ("  {0,-24} NOT self-contained" -f $header.Name)
        $output | Select-String -Pattern "error C" | Select-Object -First 3 |
            ForEach-Object { Write-Host ("      " + $_.Line.Trim()) }
        $failures++
    }
}

Write-Host ("`n{0} headers checked, {1} not self-contained" -f $checked, $failures)
exit $failures
