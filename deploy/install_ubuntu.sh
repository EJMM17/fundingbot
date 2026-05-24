#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/fundingbot"
APP_USER="fundingbot"
REPO_URL="${1:-}"

if [[ -z "$REPO_URL" ]]; then
  echo "Uso: sudo bash deploy/install_ubuntu.sh <git_repo_url>"
  echo "Ejemplo: sudo bash deploy/install_ubuntu.sh git@github.com:usuario/fundingbot.git"
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Ejecuta este script con sudo."
  exit 1
fi

apt-get update
apt-get install -y python3 python3-venv python3-pip git ca-certificates

if ! id "$APP_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone "$REPO_URL" "$APP_DIR"
else
  git -C "$APP_DIR" pull --ff-only
fi

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

if [[ ! -f "$APP_DIR/.env" ]]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "Se creo $APP_DIR/.env. Editalo con tus credenciales antes de iniciar el servicio."
fi

chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$APP_DIR/.env"

cp "$APP_DIR/deploy/fundingbot.service" /etc/systemd/system/fundingbot.service
systemctl daemon-reload
systemctl enable fundingbot.service

echo
echo "Instalacion lista."
echo "1) Edita secretos: sudo nano $APP_DIR/.env"
echo "2) Prueba imports: sudo -u $APP_USER $APP_DIR/.venv/bin/python -m compileall $APP_DIR"
echo "3) Inicia: sudo systemctl start fundingbot"
echo "4) Logs: sudo journalctl -u fundingbot -f"
