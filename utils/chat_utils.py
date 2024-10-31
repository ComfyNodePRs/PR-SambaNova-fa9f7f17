import uuid
import json
import os
import logging
from collections import OrderedDict
import time
import threading
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatHistoryManager:
    def __init__(self, history_file: str = "Nova.json", max_conversations: int = 100):
        self.history_file = self.get_history_file_path(history_file)
        self.max_conversations = max_conversations
        self.lock = threading.Lock()
        logger.info(f"Initializing ChatHistoryManager with file: {self.history_file}")
        self.ensure_valid_file()

    def get_history_file_path(self, filename: str) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while os.path.basename(current_dir) != 'SambaNova':
            current_dir = os.path.dirname(current_dir)
            if current_dir == os.path.dirname(current_dir):
                raise FileNotFoundError("Could not find 'SambaNova' directory")
        json_path = os.path.join(current_dir, 'nodes', 'Nova', filename)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        return json_path

    def ensure_valid_file(self) -> None:
        if not os.path.exists(self.history_file):
            logger.info(f"Creating new history file: {self.history_file}")
            with open(self.history_file, 'w') as f:
                json.dump({}, f)

    def load_history(self) -> OrderedDict:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self.lock, open(self.history_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        return OrderedDict(json.loads(content))
                    else:
                        logger.warning("History file is empty")
                        return OrderedDict()
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Unexpected error loading history (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(0.1)
        logger.error("Failed to load history after multiple attempts")
        return OrderedDict()

    def save_history(self, conversations: OrderedDict) -> None:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with self.lock, open(self.history_file, 'w') as f:
                    json.dump(conversations, f, indent=2)
                logger.info(f"History saved successfully. Total conversations: {len(conversations)}")
                return
            except Exception as e:
                logger.error(f"Error saving history (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(0.1)
        logger.error("Failed to save history after multiple attempts")

    def create_new_conversation(self) -> str:
        conversations = self.load_history()
        conversation_id = str(uuid.uuid4())
        conversations[conversation_id] = []
        if len(conversations) > self.max_conversations:
            oldest_conversation = next(iter(conversations))
            del conversations[oldest_conversation]
            logger.info(f"Removed oldest conversation {oldest_conversation} due to limit")
        self.save_history(conversations)
        return conversation_id

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        conversations = self.load_history()
        return conversations.get(conversation_id, [])

    def update_history(self, conversation_id: str, messages: List[Dict[str, str]]) -> None:
        conversations = self.load_history()
        conversations[conversation_id] = messages
        self.save_history(conversations)

    def get_all_conversations(self) -> OrderedDict:
        return self.load_history()

    def delete_conversation(self, conversation_id: str) -> None:
        conversations = self.load_history()
        if conversation_id in conversations:
            del conversations[conversation_id]
            self.save_history(conversations)
            logger.info(f"Deleted conversation {conversation_id}")
        else:
            logger.warning(f"Conversation {conversation_id} not found for deletion")

    def clear_all_conversations(self) -> None:
        self.save_history({})
        logger.info("Cleared all conversations")

    def get_conversation_summary(self, conversation_id: str) -> str:
        history = self.get_history(conversation_id)
        if not history:
            return "Empty conversation"
        message_count = len(history)
        last_message = history[-1]['content'][:50] + "..." if len(history[-1]['content']) > 50 else history[-1]['content']
        return f"Messages: {message_count}, Last message: {last_message}"

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        history = self.get_history(conversation_id)
        history.append({"role": role, "content": content})
        self.update_history(conversation_id, history)

    def get_token_count(self, conversation_id: str) -> int:
        history = self.get_history(conversation_id)
        # This is a simple estimation. You might want to use a proper tokenizer for accurate count
        return sum(len(message['content'].split()) for message in history)

    def truncate_history(self, conversation_id: str, max_tokens: int) -> None:
        history = self.get_history(conversation_id)
        while self.get_token_count(conversation_id) > max_tokens and history:
            history.pop(0)
        self.update_history(conversation_id, history)

    def get_last_n_messages(self, conversation_id: str, n: int) -> List[Dict[str, str]]:
        history = self.get_history(conversation_id)
        return history[-n:] if n < len(history) else history