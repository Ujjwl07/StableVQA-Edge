@echo off
REM ===================== EDIT THESE TWO LINES IF NEEDED =====================
set EXE=stablevqa.exe
set VIDEOLIST=list.txt
REM =========================================================================
echo ================================================================
echo  CONFIG 1 : P-cores only (OURS)  -- pinned, default policy
echo  (no env vars; the binary auto-selects P-cores and pins to them)
echo ================================================================
set SVQA_NO_AFFINITY=
set SVQA_THREADS=
"%EXE%" "%VIDEOLIST%" out_ours.csv
echo.
echo  ^>^> Record the "Total time" printed above for row: P-cores only (ours)
echo ================================================================
pause
