import os
import configparser
import json
import logging
from ..utils.api_utils import make_api_request, make_streaming_request
from ..utils.chat_utils import ChatHistoryManager
from ..utils.prompt_utils import load_prompt_options, format_prompt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SambaNovaLLMNode:
    SAMBA_NOVA_MODELS = [
        "Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-70B-Instruct",
        "Meta-Llama-3.1-405B-Instruct",
        "Meta-Llama-3.2-1B-Instruct",
        "Meta-Llama-3.2-3B-Instruct"
    ]

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_path = os.path.join(os.path.dirname(__file__), 'Nova', 'SambaNovaConfig.ini')
        self.load_config()
        self.chat_history_manager = ChatHistoryManager()
        self.prompt_options = load_prompt_options([
            os.path.join(os.path.dirname(__file__), 'Nova', 'DefaultPrompts.json'),
            os.path.join(os.path.dirname(__file__), 'Nova', 'UserPrompts.json')
        ])

    def load_config(self):
        if os.path.exists(self.config_path):
            self.config.read(self.config_path)
            logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Config file not found at {self.config_path}. Using default values.")
            self.config['API'] = {'key': '', 'base_url': 'https://api.sambanova.ai/v1', 'max_retries': '3'}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Enter your prompt here"}),
                "model": (cls.SAMBA_NOVA_MODELS, {"default": "Meta-Llama-3.1-8B-Instruct"}),
                "max_tokens": ("INT", {"default": 100, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 1, "min": 1, "max": 100}),
                "request_type": (["completion", "chat"], {"default": "completion"}),
            },
            "optional": {
                "system_message": ("STRING", {"multiline": True, "default": ""}),
                "stop_sequences": ("STRING", {"default": ""}),
                "conversation_id": ("STRING", {"default": ""}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.01}),
                "stream": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = ("generated_text", "token_count", "conversation_id")
    FUNCTION = "generate_text"
    CATEGORY = "LLM"

    def generate_text(self, prompt, model, max_tokens, temperature, top_p, top_k, request_type,
                      system_message="", stop_sequences="", conversation_id="",
                      repetition_penalty=1.0, stream=False):
        api_key = self.config.get('API', 'key', fallback='')
        base_url = self.config.get('API', 'base_url', fallback='https://api.sambanova.ai/v1')
        max_retries = int(self.config.get('API', 'max_retries', fallback='3'))

        if not api_key:
            raise ValueError("API key is not set in the SambaNovaConfig.ini file.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        if not conversation_id:
            conversation_id = self.chat_history_manager.create_new_conversation()
        
        conversation_history = self.chat_history_manager.get_history(conversation_id)
        
        data = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "stream": stream,
        }

        if stop_sequences:
            data["stop"] = [seq.strip() for seq in stop_sequences.split(',')]

        if request_type == "chat":
            data["messages"] = []
            if system_message:
                data["messages"].append({"role": "system", "content": system_message})
            for message in conversation_history:
                data["messages"].append(message)
            data["messages"].append({"role": "user", "content": prompt})
            endpoint = f"{base_url}/chat/completions"
        else:  # completion
            full_prompt = format_prompt(system_message, conversation_history, prompt)
            data["prompt"] = full_prompt
            endpoint = f"{base_url}/completions"

        if stream:
            generated_text, token_count = self.handle_streaming_response(data, headers, endpoint, conversation_id)
        else:
            generated_text, token_count = self.handle_non_streaming_response(data, headers, endpoint, max_retries, request_type, conversation_id, prompt)

        self.update_chat_history(conversation_id, prompt, generated_text)
        return (generated_text, token_count, conversation_id)

    def handle_streaming_response(self, data, headers, endpoint, conversation_id):
        generated_text = ""
        token_count = 0
        for chunk in make_streaming_request(data, headers, endpoint):
            if chunk.startswith("Error:"):
                logger.error(chunk)
                return chunk, 0
            generated_text += chunk
            token_count += 1  # This is still an approximation
        return generated_text, token_count

    def handle_non_streaming_response(self, data, headers, endpoint, max_retries, request_type, conversation_id, prompt):
        response, success, status_code = make_api_request(data, headers, endpoint, max_retries)

        if success:
            if request_type == "chat":
                generated_text = response["choices"][0]["message"]["content"].strip()
            else:
                generated_text = response["choices"][0]["text"].strip()

            token_count = response.get("usage", {}).get("total_tokens", 0)
            
            logger.info(f"Successfully generated text with {token_count} tokens using {data['model']}.")
            return generated_text, token_count
        else:
            error_message = f"Error: {response}"
            logger.error(error_message)
            return error_message, 0

    def update_chat_history(self, conversation_id, prompt, response):
        conversation_history = self.chat_history_manager.get_history(conversation_id)
        conversation_history.append({"role": "user", "content": prompt})
        conversation_history.append({"role": "assistant", "content": response})
        self.chat_history_manager.update_history(conversation_id, conversation_history)