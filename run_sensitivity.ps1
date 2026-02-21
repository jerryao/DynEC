$lambdas = @(0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
$root = $PSScriptRoot
$script = "$root\run_dynec_impl.py"
$data = "$root\Sichuan2024Dataset_Monthly"
$out_base = "$root\Sichuan2024_Experiments\results\sensitivity"

if (!(Test-Path -Path $out_base)) {
    New-Item -ItemType Directory -Force -Path $out_base | Out-Null
}

foreach ($lam in $lambdas) {
    Write-Output "Running with lambda=$lam..."
    $out_dir = "$out_base\lambda_$lam"
    if (!(Test-Path -Path $out_dir)) {
        New-Item -ItemType Directory -Force -Path $out_dir | Out-Null
    }
    python $script --city City-A --data-root $data --out-dir $out_dir --max-days 0 --limit-users 0 --lambda-temp $lam
}