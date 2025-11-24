@echo off
REM =====================================================
REM  SUBIR DATOS A KAGGLE - Método Actualizado
REM =====================================================

echo.
echo ====================================================
echo   SUBIENDO DATOS A KAGGLE
echo ====================================================
echo.

REM 1. Verificar que KAGGLE_API_TOKEN está configurado
if "%KAGGLE_API_TOKEN%"=="" (
    echo ERROR: KAGGLE_API_TOKEN no configurado
    echo.
    echo Por favor ejecuta primero:
    echo setx KAGGLE_API_TOKEN "TU_TOKEN_AQUI"
    echo.
    echo O en PowerShell:
    echo [System.Environment]::SetEnvironmentVariable('KAGGLE_API_TOKEN', 'TU_TOKEN', 'User'^)
    echo.
    pause
    exit /b 1
)

echo [OK] Token configurado

REM 2. Instalar Kaggle API si no está
pip show kaggle >nul 2>&1
if errorlevel 1 (
    echo [1/5] Instalando Kaggle API...
    pip install kaggle
) else (
    echo [1/5] Kaggle API ya instalado
)

REM 3. Verificar que dataset existe
if not exist "data\triple_barrier_dataset.csv" (
    echo.
    echo ERROR: No se encuentra triple_barrier_dataset.csv
    echo Por favor ejecuta primero:
    echo python triple_barrier_labeling.py
    echo.
    pause
    exit /b 1
)

echo [2/5] Dataset encontrado

REM 4. Crear carpeta temporal
if exist "kaggle_upload" rmdir /s /q kaggle_upload
mkdir kaggle_upload

echo [3/5] Preparando archivos...

REM Copiar dataset
copy "data\triple_barrier_dataset.csv" "kaggle_upload\" >nul

REM Copiar tickers
copy "good.txt" "kaggle_upload\tickers.txt" >nul

REM 5. Crear metadata
echo [4/5] Creando metadata...

(
echo {
echo   "title": "trading-triple-barrier-dataset",
echo   "id": "%USERNAME%/trading-triple-barrier-dataset",
echo   "licenses": [{"name": "CC0-1.0"}]
echo }
) > kaggle_upload\dataset-metadata.json

REM 6. Subir a Kaggle
echo [5/5] Subiendo a Kaggle...
cd kaggle_upload

kaggle datasets create -p .

if errorlevel 1 (
    echo.
    echo ERROR: Upload fallo
    echo Verifica que el token este configurado correctamente
    cd ..
    pause
    exit /b 1
)

cd ..

REM Limpiar
rmdir /s /q kaggle_upload

echo.
echo ====================================================
echo   COMPLETADO
echo ====================================================
echo.
echo Dataset disponible en:
echo https://www.kaggle.com/datasets/%USERNAME%/trading-triple-barrier-dataset
echo.
echo SIGUIENTE PASO:
echo 1. Ve a Kaggle.com
echo 2. Crea un nuevo Notebook
echo 3. Agrega tu dataset
echo 4. Copia el codigo de kaggle_train_notebook.py
echo 5. Save ^& Run All
echo.
pause
