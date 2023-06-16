#!/usr/bin/env sh

# 启动FlaskAPP
gunicorn -c config/gunicorn.conf pro_app:app
