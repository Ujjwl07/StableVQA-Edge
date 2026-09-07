@echo off
REM ===================== EDIT THESE THREE LINES =============================
set EXE=stablevqa.exe
set VIDEOLIST=list.txt
REM  NWORKERS = "logical=" number from the CPU: startup line (all logical/SMT)
set NWORKERS=12
REM =========================================================================
echo ================================================================
echo  CONFIG 4 : All logical processors  -- ORT default behaviour, no pinning
echo  workers=%NWORKERS%  (SMT threads + E-cores)
echo ================================================================
set SVQA_NO_AFFINITY=1
set SVQA_THREADS=%NWORKERS%
"%EXE%" "%VIDEOLIST%" out_logical.csv
set SVQA_NO_AFFINITY=
set SVQA_THREADS=
echo.
echo  ^>^> Record the "Total time" printed above for row: All logical (ORT default)
echo  ^>^> Check the startup line shows [no-affinity] and intra_op_threads=%NWORKERS%
echo ================================================================
pause
