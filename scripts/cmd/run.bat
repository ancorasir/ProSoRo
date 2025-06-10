@echo off
for /f "delims=" %%i in ('dir /b *.inp') do (
abaqus job=%%i cpus=1 int
)