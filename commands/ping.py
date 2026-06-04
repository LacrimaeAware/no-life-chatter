import time
import psutil
import logging

async def handle_ping(bot, message, params):
    start_time = time.time()
    logging.info("Starting ping command processing.")


    # System info retrieval
    uptime_seconds = time.time() - psutil.boot_time()
    cpu_usage = psutil.cpu_percent(interval=None)
    memory = psutil.virtual_memory()

    logging.info("System info retrieved.")

    response_time = time.time() - start_time
    response = (
        f"Pong! 🏓 Command Processing Latency: {int(response_time * 1000)}ms; "
        f"Uptime: {int(uptime_seconds // 3600)}h {int((uptime_seconds % 3600) // 60)}m; "
        f"CPU Usage: {cpu_usage}%; Memory Usage: {memory.percent}% of {round(memory.total / (1024 ** 3), 2)}GB."
    )

    # Before sending response
    pre_send_time = time.time()
    await message.channel.send(response)
    post_send_time = time.time()

    logging.info(f"Response prepared in {int((pre_send_time - start_time) * 1000)}ms, sent in {int((post_send_time - pre_send_time) * 1000)}ms.")
