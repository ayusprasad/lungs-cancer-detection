@echo off
cd /d "%~dp0"
start "Lung Cancer Dashboard" http://127.0.0.1:5000
python server.py
