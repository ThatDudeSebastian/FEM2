@echo off
set LOCALHOST=%COMPUTERNAME%
if /i "%LOCALHOST%"=="Sebastians-Rechenzentrum" (taskkill /f /pid 29292)
if /i "%LOCALHOST%"=="Sebastians-Rechenzentrum" (taskkill /f /pid 11640)
if /i "%LOCALHOST%"=="Sebastians-Rechenzentrum" (taskkill /f /pid 17168)
if /i "%LOCALHOST%"=="Sebastians-Rechenzentrum" (taskkill /f /pid 8236)

del /F cleanup-ansys-Sebastians-Rechenzentrum-8236.bat
