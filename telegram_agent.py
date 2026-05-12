import requests
from decouple import config
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configuration
TELEGRAM_BOT_TOKEN = config('TELEGRAM_BOT_TOKEN')
DJANGO_API_BASE = "http://127.0.0.1:8000"
ALLOWED_IDS = [int(x) for x in config('TELEGRAM_ALLOWED_IDS', default='').split(',') if x.strip()]

SYSTEM_PROMPT = """
Tu es SafeBot Agent, l'assistant mobile de la plateforme SafeVision. 
Tu es expert en sécurité industrielle en Tunisie.
Tu aides les ingénieurs sur le terrain. Tu es concis, professionnel et tu réagis vite.
"""

# ── Sécurité (Whitelist) ──

def restricted(func):
    """Décorateur pour n'autoriser que les IDs dans la whitelist."""
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id not in ALLOWED_IDS:
            await update.message.reply_text("🚫 Accès refusé. Vous n'êtes pas autorisé à utiliser ce bot.")
            print(f"[SafeBot Agent] Tentative d'accès non autorisé de l'ID: {user_id}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# ── Clavier Interactif ──

def get_main_keyboard():
    keyboard = [
        [KeyboardButton("📊 Rapport du Jour"), KeyboardButton("🩺 Statut Système")],
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# ── Commandes ──

@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🛡️ Bienvenue SafeVision Agent !\n\n"
        "Utilise les boutons ci-dessous ou pose-moi une question.",
        reply_markup=get_main_keyboard()
    )

@restricted
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Système SafeVision en ligne. Modèles chargés.", reply_markup=get_main_keyboard())

@restricted
async def rapport(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # On envoie un message d'attente
    await update.message.reply_text("📊 Génération du rapport quotidien...")
    
    try:
        res = requests.get(f"{DJANGO_API_BASE}/api/daily-report/")
        if res.status_code == 200:
            data = res.json()
            text = (
                f"📊 *Rapport du {data['date']}*\n\n"
                f"🔥 Alertes Critiques: {data['critical']}\n"
                f"⚠️ Alertes Warning: {data['warning']}\n"
                f"🔢 Total: {data['total_alerts']}\n\n"
            )
            if data.get('last_alerts'):
                text += "_Dernières alertes:_\n" + "\n".join(data['last_alerts'])
            else:
                text += "Aucune alerte aujourd'hui."
                
            # CORRECTION : On envoie un NOUVEAU message avec le clavier, on n'édite pas l'ancien
            await update.message.reply_text(text, parse_mode='Markdown', reply_markup=get_main_keyboard())
        else:
            await update.message.reply_text("❌ Erreur lors de la récupération.", reply_markup=get_main_keyboard())
    except Exception as e:
        await update.message.reply_text(f"❌ Erreur de connexion: {e}", reply_markup=get_main_keyboard())

# ── Chat classique (Texte) ──

@restricted
async def chat_with_agent(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        res = requests.post(f"{DJANGO_API_BASE}/api/chat/", json={"message": user_message})
        if res.status_code == 200:
            bot_reply = res.json().get('reply', "Erreur de format.")
        else:
            bot_reply = "Le serveur Django a rencontré une erreur."
    except requests.exceptions.ConnectionError:
        bot_reply = "❌ Je n'arrive pas à joindre Django."

    await update.message.reply_text(bot_reply, reply_markup=get_main_keyboard())


# ── Lancement du Bot ──

def main():
    print("[SafeBot Agent] Démarrage du bot Telegram...")
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.Regex("^📊 Rapport du Jour$"), rapport))
    app.add_handler(MessageHandler(filters.Regex("^🩺 Statut Système$"), status))
    
    # Text handler (doit être en dernier)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_with_agent))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()