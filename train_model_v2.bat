@echo off
REM ============================================================================
REM SCRIPT DE ENTRENAMIENTO MODELO v2.0 - Trend Following
REM ============================================================================

echo ========================================================================
echo   ENTRENAMIENTO MODELO ML v2.0 - TREND FOLLOWING
echo ========================================================================
echo.
echo Este script entrena el modelo con:
echo   - Datos completos 2015-2024
echo   - Optimizacion Bayesiana (Optuna)
echo   - Triple Barrier Labels mejorado
echo   - Purged K-Fold Validation
echo.
echo Opciones:
echo   [1] Entrenamiento RAPIDO (sin optimizacion, parametros por defecto)
echo   [2] Entrenamiento OPTIMIZADO (Optuna, 50 trials, ~30-60 min)
echo   [3] Entrenamiento EXHAUSTIVO (Optuna, 100 trials, ~1-2 horas)
echo.

set /p choice="Selecciona una opcion (1/2/3): "

if "%choice%"=="1" goto QUICK
if "%choice%"=="2" goto OPTIMIZED
if "%choice%"=="3" goto EXHAUSTIVE

echo Opcion invalida. Saliendo...
goto END

:QUICK
echo.
echo ========================================================================
echo   ENTRENAMIENTO RAPIDO
echo ========================================================================
echo.
python train_signal_model_v2.py ^
    --ticker-file good.txt ^
    --output models/trend_model_2015_2024_quick.joblib ^
    --train-from 2015-01-01 ^
    --train-until 2022-12-31 ^
    --horizon 10 ^
    --atr-multiplier-tp 3.0 ^
    --atr-multiplier-sl 2.0 ^
    --model-type xgb ^
    --n-estimators 300 ^
    --max-depth 8 ^
    --learning-rate 0.05 ^
    --subsample 0.8 ^
    --colsample-bytree 0.8 ^
    --class-weighted ^
    --n-splits 5 ^
    --purge-window 10 ^
    --shap-report-dir reports/shap_v2_quick

goto END

:OPTIMIZED
echo.
echo ========================================================================
echo   ENTRENAMIENTO OPTIMIZADO (Optuna 50 trials)
echo ========================================================================
echo.
python train_signal_model_v2.py ^
    --ticker-file good.txt ^
    --output models/trend_model_2015_2024_optimized.joblib ^
    --train-from 2015-01-01 ^
    --train-until 2022-12-31 ^
    --horizon 10 ^
    --atr-multiplier-tp 3.0 ^
    --atr-multiplier-sl 2.0 ^
    --model-type xgb ^
    --optimize-hyperparams ^
    --n-trials 50 ^
    --class-weighted ^
    --n-splits 5 ^
    --purge-window 10 ^
    --shap-report-dir reports/shap_v2_optimized

goto END

:EXHAUSTIVE
echo.
echo ========================================================================
echo   ENTRENAMIENTO EXHAUSTIVO (Optuna 100 trials)
echo ========================================================================
echo.
python train_signal_model_v2.py ^
    --ticker-file good.txt ^
    --output models/trend_model_2015_2024_exhaustive.joblib ^
    --train-from 2015-01-01 ^
    --train-until 2022-12-31 ^
    --horizon 10 ^
    --atr-multiplier-tp 3.0 ^
    --atr-multiplier-sl 2.0 ^
    --model-type xgb ^
    --optimize-hyperparams ^
    --n-trials 100 ^
    --class-weighted ^
    --n-splits 5 ^
    --purge-window 10 ^
    --shap-report-dir reports/shap_v2_exhaustive

goto END

:END
echo.
echo ========================================================================
echo   ENTRENAMIENTO COMPLETADO
echo ========================================================================
echo.
pause
