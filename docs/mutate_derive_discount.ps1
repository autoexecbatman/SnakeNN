# Mutation test for az::deriveDiscount.
#
# Its properties are static_asserts, so every kill here is a compile error rather than
# a failing run. That is not a weaker result: a static_assert that does not fire is a
# claim the compiler is not checking, and this is what shows each one bites.
#
# The mutant that matters is `reproduces_paper`: the rule's whole justification is that
# it returns exactly 0.98 at 10x10, so any change to the constant must break that.
#
# Run it:
#
#     powershell -NoProfile -ExecutionPolicy Bypass -File docs/mutate_derive_discount.ps1
#
# Needs cl on PATH. Prints one line per mutant and a count.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants_derive_discount"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/az_parameters.h" -Raw -Encoding UTF8

$mutants = @(
  # The halving is what makes 10x10 come out at the paper's 0.98. Any other divisor
  # breaks that anchor, which is the one property the rule rests on.
  @{ name = "no_halving";        from = "static_cast<float>(board) / 2.0f"; to = "static_cast<float>(board)" },
  @{ name = "quarter_horizon";   from = "static_cast<float>(board) / 2.0f"; to = "static_cast<float>(board) / 4.0f" },
  # Linear in the side rather than in the area. This is the plausible wrong shape:
  # it still grows with the board and still returns something sane-looking.
  @{ name = "linear_in_side";    from = "const float horizon = static_cast<float>(board) * static_cast<float>(board) / 2.0f;"; to = "const float horizon = static_cast<float>(board) / 2.0f;" },
  # A discount of one never stops summing; a negative one is not a discount at all.
  @{ name = "returns_one";       from = "return 1.0f - 1.0f / horizon;"; to = "return 1.0f;" },
  @{ name = "sign_flipped";      from = "return 1.0f - 1.0f / horizon;"; to = "return 1.0f + 1.0f / horizon;" },
  # Constant, which is exactly the behaviour the rule replaced.
  @{ name = "back_to_constant";  from = "return 1.0f - 1.0f / horizon;"; to = "return DISCOUNT;" }
)

$killed = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    foreach ($file in @("az_parameters_test.cpp", "snake_env.h", "small_random.h", "snake_logic.h"))
    {
        Copy-Item "$src/$file" $dir
    }
    Copy-Item "$src/az_parameters.cpp" $dir
    Copy-Item "$src/snake_env.cpp" $dir

    if (-not $original.Contains($mutant.from))
    {
        Write-Host ("  {0,-22} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $mutated = $original.Replace($mutant.from, $mutant.to)
    Set-Content "$dir/az_parameters.h" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    $null = cl /nologo /std:c++20 /EHsc /O2 az_parameters_test.cpp az_parameters.cpp snake_env.cpp /Fe:mutant.exe 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-22} KILLED by a static_assert" -f $mutant.name)
        $killed++
        Pop-Location
        continue
    }
    $null = & ./mutant.exe 2>&1
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Write-Host ("  {0,-22} KILLED at run time" -f $mutant.name); $killed++ }
    else { Write-Host ("  {0,-22} SURVIVED - nothing checks this" -f $mutant.name) }
}
Write-Host ("`nkilled {0} of {1}" -f $killed, $mutants.Count)
