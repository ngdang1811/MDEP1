@echo off
echo ==========================================================
echo  GUDS-EDL Automated Benchmark Suite
echo ==========================================================

REM Ensure Python is available
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b
)

echo.
echo [1/3] Running CIFAR-100-LT Benchmarks...
echo ----------------------------------------
python cifar_lt_runner.py --imbalance_ratio 10 --epochs 100
python cifar_lt_runner.py --imbalance_ratio 50 --epochs 100
python cifar_lt_runner.py --imbalance_ratio 100 --epochs 100

echo.
echo [2/3] Running MVTec AD Benchmarks...
echo ----------------------------------------
python mvtec_ad_runner.py --category hazelnut --epochs 20
python mvtec_ad_runner.py --category bottle --epochs 20

echo.
echo [3/3] Running Main ISIC 2024 Benchmark...
echo ----------------------------------------
python ..\guds_edl_core.py

echo.
echo ==========================================================
echo  All Benchmarks Completed!
echo  To run the comprehensive Component Ablation Study,
echo  run: python isic_paper_experiments.py --suite ablations --seeds 42 43 44 --no_save_model
echo ==========================================================
pause
