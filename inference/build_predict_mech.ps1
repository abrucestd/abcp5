$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$ortRoot = "C:\onnxruntime"
$include = Join-Path $ortRoot "include"
$lib = Join-Path $ortRoot "lib\onnxruntime.lib"

g++ -std=c++17 -O2 .\predict_mech.cpp `
    "-I$include" `
    $lib `
    -static `
    -static-libgcc `
    -static-libstdc++ `
    -o abcp5.5.exe

# Keep the conventional name for callers that expect the abcp5 layout.
Copy-Item .\abcp5.5.exe .\abcp5.exe -Force

g++ -std=c++17 -O2 .\result_process4.cpp `
    -static `
    -static-libgcc `
    -static-libstdc++ `
    -o result_process4.exe

Copy-Item -Path (Join-Path $ortRoot "lib\onnxruntime.dll") -Destination .\onnxruntime.dll -Force

Write-Host "[INFO] built abcp5.5.exe"
Write-Host "[INFO] copied compatibility alias abcp5.exe"
Write-Host "[INFO] built result_process4.exe"
Write-Host "[INFO] copied onnxruntime.dll"
