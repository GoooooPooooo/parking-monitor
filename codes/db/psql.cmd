@echo off
chcp 65001 >nul
set PGCLIENTENCODING=UTF8
set PGPASSWORD=postgres
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -d parking_monitor %*
