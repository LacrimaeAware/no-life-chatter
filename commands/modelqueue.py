from auth import is_super_admin
from services import model_queue

description = (
    "Inspect or clear the shared local-model queue (super admins).\n"
    "  ~modelqueue [status|clear]"
)


async def handle_modelqueue(bot, message, params):
    if not message.author or not is_super_admin(message.author.name):
        return
    action = (params[0].lower() if params else "status")
    if action in {"clear", "flush", "cancel"}:
        cleared = model_queue.clear_pending()
        running = " Running command will finish." if model_queue.running_ticket() else ""
        await message.channel.send(f"model queue cleared: {cleared} pending.{running}")
        return
    if action != "status":
        await message.channel.send("Usage: ~modelqueue [status|clear]")
        return
    await message.channel.send(model_queue.status())
