# Configure + build with MSVC (Visual Studio Build Tools) and Ninja.
# Usage:  .\build.ps1 [-Target cetk_cli|cetk_fixgw|cetk_tests|all] [-Config Release|Debug]
param([string]$Target = "all", [string]$Config = "Release")
$ErrorActionPreference = "Stop"
$vs = Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $vs) { throw "vcvars64.bat not found - install Visual Studio Build Tools (C++ workload)" }
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) { $cmake = Join-Path (python -c "import cmake,os;print(cmake.CMAKE_BIN_DIR)") "cmake.exe" }
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) { $ninja = Join-Path (python -c "import ninja;print(ninja.BIN_DIR)") "ninja.exe" }
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$build = Join-Path $root "build"
$tgt = if ($Target -eq "all") { "" } else { "--target $Target" }
cmd /c "`"$($vs.FullName)`" >nul 2>&1 && `"$cmake`" -S `"$root`" -B `"$build`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=$Config && `"$cmake`" --build `"$build`" $tgt"
exit $LASTEXITCODE
