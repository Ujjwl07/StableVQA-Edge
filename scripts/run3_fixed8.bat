@echo off
REM ===================== EDIT THESE TWO LINES IF NEEDED =====================
set EXE=stablevqa.exe
set VIDEOLIST=list.txt
REM =========================================================================
echo ================================================================
echo  CONFIG 3 : Fixed constant  -- 8 threads, no pinning (the old default)
echo ================================================================
set SVQA_NO_AFFINITY=1
set SVQA_THREADS=8
"%EXE%" "%VIDEOLIST%" out_fixed8.csv
set SVQA_NO_AFFINITY=
set SVQA_THREADS=
echo.
echo  ^>^> Record the "Total time" printed above for row: Fixed constant (8)
echo ================================================================
pause
