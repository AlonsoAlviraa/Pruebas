@echo off
REM =====================================================
REM  COMPRIMIR DATOS PARA KAGGLE
REM =====================================================

echo.
echo ====================================================
echo   COMPRIMIENDO DATOS PARA KAGGLE
echo ====================================================
echo.

REM 1. Verificar que existe carpeta data
if not exist "data\" (
    echo ERROR: Carpeta data\ no encontrada
    pause
    exit /b 1
)

REM 2. Crear carpeta temporal
if exist "kaggle_upload" rmdir /s /q kaggle_upload
mkdir kaggle_upload

echo [1/3] Contando archivos...

REM Contar archivos CSV
set count=0
for /r "data\" %%f in (*.csv) do set /a count+=1
echo   Encontrados: %count% archivos CSV

REM 3. Comprimir con PowerShell (más rápido)
echo [2/3] Comprimiendo datos...
echo   Esto puede tardar 2-3 minutos...

powershell -Command "Compress-Archive -Path 'data\*_history.csv' -DestinationPath 'kaggle_upload\trading_data.zip' -Force"

if errorlevel 1 (
    echo.
    echo ERROR: Compresion fallo
    pause
    exit /b 1
)

REM Verificar tamaño
for %%A in ("kaggle_upload\trading_data.zip") do set size=%%~zA
set /a sizeMB=%size% / 1048576

echo   Archivo ZIP creado: %sizeMB% MB

REM 4. Crear metadata
echo [3/3] Creando metadata...

(
echo {
echo   "title": "trading-raw-data-compressed",
echo   "id": "%USERNAME%/trading-raw-data-compressed",
echo   "licenses": [{"name": "CC0-1.0"}]
echo }
) > kaggle_upload\dataset-metadata.json

echo.
echo ====================================================
echo   COMPLETADO
echo ====================================================
echo.
echo Archivo ZIP creado: kaggle_upload\trading_data.zip
echo Tamaño: %sizeMB% MB
echo.
echo SIGUIENTE PASO:
echo 1. Ve a https://www.kaggle.com/datasets
echo 2. Click "New Dataset"
echo 3. Arrastra el archivo: kaggle_upload\trading_data.zip
echo 4. Title: "trading-raw-data-compressed"
echo 5. Click "Create"
echo.
echo ALTERNATIVA (CLI):
echo cd kaggle_upload
echo kaggle datasets create -p .
echo.
pause
