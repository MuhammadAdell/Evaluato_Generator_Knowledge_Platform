#!/bin/bash
cd "$(dirname "$0")"

REDIS=./third_party/redis-stable/src

if [ ! -x "$REDIS/redis-server" ]; then
  echo "Redis is not built yet. Run the setup in README.md first:"
  echo "  mkdir -p third_party && cd third_party"
  echo "  curl -L https://download.redis.io/redis-stable.tar.gz | tar xz"
  echo "  cd redis-stable && make -j4 MALLOC=libc"
  exit 1
fi

if ! $REDIS/redis-cli ping > /dev/null 2>&1; then
  $REDIS/redis-server --port 6379 --daemonize yes \
    --dir "$PWD/third_party" --logfile "$PWD/third_party/redis.log"
  sleep 1
fi

.venv/bin/python -m streamlit run app.py