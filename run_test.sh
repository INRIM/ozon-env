#!/bin/bash
echo "update system"
poetry update
echo "make compose"
docker compose  up -d --force-recreate
echo "check code"
poetry run black ozonenv/**/*.py
#poetry run flake8 ozonenv/**/*.py
pip install --upgrade -e .
echo "run test"
rm -rf tests/models
time poetry run pytest --cov --cov-report=html -vv -x -s
docker compose down
echo "make project: Done."
