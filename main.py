# main.py
import os
import uvicorn
import logging

# Creating logger
dev_logger = logging.getLogger('chatbot_logger')
dev_logger.setLevel(logging.INFO)

# Create a handler and associate the formatter with it
formatter = logging.Formatter('%(asctime)s  | %(levelname)s | %(filename)s | %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)

dev_logger.addHandler(handler)

logger = logging.getLogger('chatbot_logger')

def main():
    """Entry point"""
    logger.info("Starting Medical-Chatbot Service app...")
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("src.api:app", host="0.0.0.0", port=8080, reload=True)

if __name__ == "__main__":
    main()
