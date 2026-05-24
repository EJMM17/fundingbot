# Deploy En DigitalOcean

Guia para subir este bot a un repo privado de GitHub y correrlo en un Droplet Ubuntu con `systemd`.

## 1. Preparar GitHub

1. Crea un repositorio **privado** en GitHub.
2. Desde esta carpeta:

```bash
git init
git add .
git commit -m "Prepare funding bot deploy"
git branch -M main
git remote add origin git@github.com:TU_USUARIO/fundingbot.git
git push -u origin main
```

Si Git pide identidad para el commit:

```bash
git config user.name "Tu Nombre"
git config user.email "tu-email@ejemplo.com"
```

Antes de hacer `git add .`, verifica que no aparezcan secretos:

```bash
git status --short
```

No deben subirse `.env`, `bot.db`, `bot.log`, `.venv/` ni `__pycache__/`.

## 2. Crear Droplet

Recomendado:

- Ubuntu 24.04 LTS.
- 1 vCPU / 1 GB RAM como punto de partida.
- Autenticacion por SSH key.
- Firewall con SSH limitado a tu IP si es posible.

Entra al servidor:

```bash
ssh root@IP_DEL_DROPLET
```

## 3. Instalar El Bot

Clona o copia el repo en el servidor y ejecuta el instalador:

```bash
apt-get update
apt-get install -y git
git clone git@github.com:TU_USUARIO/fundingbot.git /tmp/fundingbot
cd /tmp/fundingbot
sudo bash deploy/install_ubuntu.sh git@github.com:TU_USUARIO/fundingbot.git
```

Si usas HTTPS en vez de SSH:

```bash
sudo bash deploy/install_ubuntu.sh https://github.com/TU_USUARIO/fundingbot.git
```

## 4. Configurar `.env`

Edita el archivo real en el servidor:

```bash
sudo nano /opt/fundingbot/.env
```

Minimo recomendado al inicio:

```env
DRY_RUN=true
KUCOIN_API_KEY=...
KUCOIN_SECRET=...
KUCOIN_PASSPHRASE=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_ENABLED=true
ENABLE_AUTO_LEARNING=true
```

Para obtener `TELEGRAM_CHAT_ID`, crea un bot con `@BotFather`, escribele un mensaje y visita:

```text
https://api.telegram.org/botTU_TOKEN/getUpdates
```

## 5. Probar Antes De Arrancar

```bash
cd /opt/fundingbot
sudo -u fundingbot .venv/bin/python -m compileall .
sudo -u fundingbot .venv/bin/python -m unittest -v
```

Prueba arranque manual en dry-run:

```bash
sudo -u fundingbot .venv/bin/python main.py
```

Detenlo con `Ctrl+C` cuando veas que conecta y Telegram responde.

## 6. Correr Con systemd

```bash
sudo systemctl start fundingbot
sudo systemctl status fundingbot --no-pager
sudo journalctl -u fundingbot -f
```

Comandos utiles:

```bash
sudo systemctl restart fundingbot
sudo systemctl stop fundingbot
sudo systemctl enable fundingbot
```

## 7. Actualizar Version

```bash
cd /opt/fundingbot
sudo systemctl stop fundingbot
sudo -u fundingbot git pull --ff-only
sudo -u fundingbot .venv/bin/pip install -r requirements.txt
sudo -u fundingbot .venv/bin/python -m unittest -v
sudo systemctl start fundingbot
```

`requirements.txt` usa versiones fijas para que el Droplet corra las mismas dependencias directas que fueron probadas localmente.

## Checklist Antes De Operar Real

- `DRY_RUN=true` probado varias horas.
- Telegram responde a `/ping`, `/status`, `/positions`, `/risk` y `/learn`.
- KuCoin API key con permisos minimos necesarios.
- IP whitelist activa en KuCoin.
- `MAX_TOTAL_MARGIN`, `MAX_MARGIN_PER_COIN` y `MAX_OPEN_POSITIONS` revisados.
- `bot.db` respaldado si te importa conservar el aprendizaje local.
- Cambiar `DRY_RUN=false` solo cuando los logs sean estables.
