@echo off
set TIMESTAMP=%date:~0,10%-%time:~0,8%
set TIMESTAMP=%TIMESTAMP: =0%
set TIMESTAMP=%TIMESTAMP::=-%
python main.py --phase test > test_%TIMESTAMP%.log