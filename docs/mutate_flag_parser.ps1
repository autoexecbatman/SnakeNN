# Mutation test for flags::parseWholeInt.
#
# Each mutant is compiled in its own directory together with a copy of the header
# and the test, because MSVC resolves a quoted include from the including file's
# own directory first - a mutant left elsewhere is silently ignored and reports as
# surviving.

$src = "D:/repo/snakeNN/src"
$work = "$env:TEMP/mutants"
Remove-Item $work -Recurse -Force -ErrorAction Ignore
New-Item -ItemType Directory -Force $work | Out-Null

$original = Get-Content "$src/flag_parser.cpp" -Raw -Encoding UTF8

$mutants = @(
  @{ name = "drop_full_consumption"; from = "result.ec != std::errc{} || result.ptr != text.data() + text.size()"; to = "result.ec != std::errc{}" },
  @{ name = "drop_errc_check";       from = "result.ec != std::errc{} || result.ptr != text.data() + text.size()"; to = "result.ptr != text.data() + text.size()" },
  @{ name = "drop_range_branch";     from = "result.ec == std::errc::result_out_of_range";   to = "false" },
  @{ name = "off_by_one_return";     from = "return number;";                                 to = "return number + 1;" },
  @{ name = "always_zero";           from = "return number;";                                 to = "return 0;" },
  @{ name = "message_drops_flag";    from = '"{} needs a whole number, got ''{}''", flag, text'; to = '"needs a whole number, got ''{}''", text' },
  # The two functions share a body shape, so a plain substring hits the wrong one.
  # `scope` restricts the replacement to the text from that marker onward.
  @{ name = "u_signed_accumulator";  from = "unsigned int number = 0;"; to = "int number = 0;" },
  @{ name = "u_drop_consumption";    scope = "unsigned int parseWholeUnsigned"; from = "result.ec != std::errc{} || result.ptr != text.data() + text.size()"; to = "false" },
  @{ name = "u_drop_range_branch";   scope = "unsigned int parseWholeUnsigned"; from = "result.ec == std::errc::result_out_of_range"; to = "false" },
  @{ name = "u_always_zero";         scope = "unsigned int parseWholeUnsigned"; from = "return number;"; to = "return 0;" },
  # parseUnitFloat. Scoped, because its body shares its shape with the two above and an
  # unscoped replacement would mutate all three at once and report a kill for the wrong
  # reason.
  @{ name = "unit_drop_consumption";    scope = "float parseUnitFloat"; from = "result.ec != std::errc{} || result.ptr != text.data() + text.size()"; to = "result.ec != std::errc{}" },
  @{ name = "unit_drop_range_branch";   scope = "float parseUnitFloat"; from = "result.ec == std::errc::result_out_of_range"; to = "false" },
  # The unit bound. Written as a negated range test so a NaN - which compares false
  # against everything - is refused; the two mutants below are the two ways to lose that.
  @{ name = "unit_no_bound";       scope = "float parseUnitFloat"; from = "    if (!(number >= 0.0f && number <= 1.0f))"; to = "    if (false)" },
  @{ name = "unit_bound_admits_nan";    scope = "float parseUnitFloat"; from = "    if (!(number >= 0.0f && number <= 1.0f))"; to = "    if (number < 0.0f || number > 1.0f)" },
  # The endpoints are legal: zero is off and one is always. Excluding them turns a
  # deliberate setting into an error.
  @{ name = "unit_bound_excludes_ends"; scope = "float parseUnitFloat"; from = "    if (!(number >= 0.0f && number <= 1.0f))"; to = "    if (!(number > 0.0f && number < 1.0f))" },
  @{ name = "unit_upper_bound_only";    scope = "float parseUnitFloat"; from = "number >= 0.0f && number <= 1.0f"; to = "number <= 1.0f" },
  @{ name = "unit_lower_bound_only";    scope = "float parseUnitFloat"; from = "number >= 0.0f && number <= 1.0f"; to = "number >= 0.0f" },
  @{ name = "u_message_drops_flag";  from = '"{} needs a whole number that is not negative, got ''{}''", flag, text'; to = '"needs a whole number that is not negative, got ''{}''", text' },
  # requireAtLeast. The two off-by-one mutants are what the paired bounds in the
  # test exist to catch; the never-throws one is what the accept cases cannot.
  @{ name = "r_never_throws";        from = "if (value < minimum)"; to = "if (false)" },
  @{ name = "r_always_throws";       from = "if (value < minimum)"; to = "if (true)" },
  @{ name = "r_off_by_one_strict";   from = "if (value < minimum)"; to = "if (value <= minimum)" },
  @{ name = "r_off_by_one_loose";    from = "if (value < minimum)"; to = "if (value < minimum - 1)" },
  @{ name = "r_reversed";            from = "if (value < minimum)"; to = "if (minimum < value)" },
  @{ name = "r_message_swaps_args";  from = '"{} must be at least {}, got {}", flag, minimum, value'; to = '"{} must be at least {}, got {}", flag, value, minimum' },
  # readFlags.
  @{ name = "f_drop_flag_prefix";    from = 'if (!flag.starts_with("--"))'; to = "if (false)" },
  @{ name = "f_single_dash_ok";      from = 'flag.starts_with("--")'; to = 'flag.starts_with("-")' },
  @{ name = "f_drop_missing_value";  from = "if (index + 1 >= arguments.size())"; to = "if (false)" },
  @{ name = "f_drop_midline_guard";  from = 'if (value.starts_with("--"))'; to = "if (false)" },
  @{ name = "f_step_by_one";         from = "index += 2"; to = "index += 1" },
  @{ name = "f_swap_flag_and_value"; from = "pairs.push_back(FlagValue{ flag, value });"; to = "pairs.push_back(FlagValue{ value, flag });" },
  @{ name = "f_drop_last_pair";      from = "index < arguments.size(); index += 2"; to = "index + 2 < arguments.size(); index += 2" }
)

$killed = 0
foreach ($mutant in $mutants)
{
    $dir = Join-Path $work $mutant.name
    New-Item -ItemType Directory -Force $dir | Out-Null
    Copy-Item "$src/flag_parser.h" $dir
    Copy-Item "$src/flag_parser_test.cpp" $dir

    # Without a scope the replacement applies to the whole file; with one it applies
    # only from the marker onward, so a shape shared by two functions can be
    # mutated in exactly one of them.
    $head = ""
    $body = $original
    if ($mutant.ContainsKey("scope"))
    {
        $at = $original.IndexOf($mutant.scope)
        if ($at -lt 0)
        {
            Write-Host ("  {0,-24} NOT APPLIED - scope marker absent" -f $mutant.name)
            continue
        }
        $head = $original.Substring(0, $at)
        $body = $original.Substring($at)
    }
    if (-not $body.Contains($mutant.from))
    {
        Write-Host ("  {0,-24} NOT APPLIED - pattern absent, mutation is vacuous" -f $mutant.name)
        continue
    }
    $mutated = $head + $body.Replace($mutant.from, $mutant.to)
    Set-Content "$dir/flag_parser.cpp" -Value $mutated -Encoding UTF8 -NoNewline

    Push-Location $dir
    $compile = cl /nologo /std:c++20 /EHsc /O2 flag_parser.cpp flag_parser_test.cpp /Fe:mutant.exe 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0)
    {
        Write-Host ("  {0,-24} KILLED by the compiler" -f $mutant.name)
        $killed++
        Pop-Location
        continue
    }
    $null = & ./mutant.exe 2>&1
    $code = $LASTEXITCODE
    Pop-Location
    if ($code -ne 0) { Write-Host ("  {0,-24} KILLED" -f $mutant.name); $killed++ }
    else { Write-Host ("  {0,-24} SURVIVED - the test cannot see this change" -f $mutant.name) }
}
Write-Host ("`nkilled {0} of {1}" -f $killed, $mutants.Count)
