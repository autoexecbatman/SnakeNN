# Core Guidelines analysis for the files given, or for the AlphaZero stack's
# LibTorch-free sources by default.
#
# Run from a shell with MSVC on PATH. Sources are copied to a scratch directory
# first so the object files do not land in the tree.
#
#   powershell -NoProfile -File docs/analyze.ps1
#   powershell -NoProfile -File docs/analyze.ps1 src/az_parameters.cpp

param([string[]]$Sources = @(
    "src/az_parameters.cpp", "src/az_parameters_test.cpp",
    "src/flag_parser.cpp", "src/flag_parser_test.cpp"))

$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $env:TEMP "snakeNN_analyze"
if (Test-Path $work) { Remove-Item $work -Recurse -Force }
New-Item -ItemType Directory -Force $work | Out-Null

# Headers travel with the sources: MSVC resolves a quoted include from the
# including file's own directory first.
Get-ChildItem (Join-Path $root "src") -Filter *.h | Copy-Item -Destination $work
foreach ($source in $Sources) { Copy-Item (Join-Path $root $source) $work }
$names = ($Sources | ForEach-Object { Split-Path $_ -Leaf }) -join " "

# Checks whose only remedy is the Guidelines Support Library, which this project
# does not depend on. Disabled rather than triaged by hand every run: a list that
# has to be read past to reach the real findings is a list that stops being read.
#
#   C26446 gsl::at instead of operator[]
#   C26472 gsl::narrow_cast instead of static_cast
#   C26481 no pointer arithmetic - also unavoidable, std::from_chars has no
#          string or string_view overload in C++20
#   C26821 gsl::span instead of std::span
$gsl_checks = "26446", "26472", "26481", "26821"
$disabled = ($gsl_checks | ForEach-Object { "/wd$_" }) -join " "

# /external:W0 and /analyze:external- keep the standard library's own warnings
# out; without them the real findings are buried under hundreds of them.
$command = "cd /d `"$work`" && cl /nologo /std:c++20 /EHsc /permissive- /W4 $disabled " +
           "/external:anglebrackets /external:W0 /analyze:external- " +
           "/analyze /analyze:plugin EspXEngine.dll /c $names"
$output = cmd /c $command 2>&1 | Out-String

$findings = $output -split "`r?`n" | Where-Object { $_ -match "warning C|error C" }
if ($findings)
{
    $findings | ForEach-Object { Write-Host $_ }
    Write-Host ("`n{0} findings" -f $findings.Count)
    exit 1
}
Write-Host "no findings"
exit 0
