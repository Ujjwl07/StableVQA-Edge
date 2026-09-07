@echo off
REM ===================== EDIT THESE THREE LINES =============================
set EXE=stablevqa.exe
set VIDEOLIST=list.txt
REM  NWORKERS = "physical=" number from the CPU: startup line (all physical cores)
set NWORKERS=8
REM =========================================================================
echo ================================================================
echo  CONFIG 2 : All physical cores  P+E  -- no pinning
echo  workers=%NWORKERS%  (E-cores allowed to participate)
echo ================================================================
set SVQA_NO_AFFINITY=1
set SVQA_THREADS=%NWORKERS%
"%EXE%" "%VIDEOLIST%" out_pe.csv
set SVQA_NO_AFFINITY=
set SVQA_THREADS=
echo.
echo  ^>^> Record the "Total time" printed above for row: All physical (P+E)
echo  ^>^> Check the startup line shows [no-affinity] and intra_op_threads=%NWORKERS%
echo ================================================================
pause
