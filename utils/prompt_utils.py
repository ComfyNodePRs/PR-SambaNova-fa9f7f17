import json
import os
import logging
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_prompt_options(prompt_files: List[str]) -> Dict[str, str]:
    prompt_options = {}
    for json_file in prompt_files:
        try:
            with open(json_file, 'r') as file:
                prompts = json.load(file)
                for prompt in prompts:
                    if isinstance(prompt, dict) and 'name' in prompt and 'content' in prompt:
                        prompt_options[prompt['name']] = prompt['content']
                    else:
                        logger.warning(f"Skipping invalid prompt in {json_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON in {json_file}: {str(e)}")
        except IOError as e:
            logger.error(f"Failed to read file {json_file}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading prompts from {json_file}: {str(e)}")
    return prompt_options

def get_prompt_content(prompt_options: Dict[str, str], prompt_name: str) -> str:
    content = prompt_options.get(prompt_name)
    if content is None:
        logger.warning(f"No content found for prompt: {prompt_name}")
        return "No content found for selected prompt"
    return content

def save_prompt(prompt_file: str, prompt_name: str, prompt_content: str) -> bool:
    try:
        prompts = []
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as file:
                prompts = json.load(file)
        
        # Update existing prompt or add new one
        updated = False
        for prompt in prompts:
            if prompt['name'] == prompt_name:
                prompt['content'] = prompt_content
                updated = True
                break
        if not updated:
            prompts.append({'name': prompt_name, 'content': prompt_content})
        
        with open(prompt_file, 'w') as file:
            json.dump(prompts, file, indent=2)
        logger.info(f"Saved prompt '{prompt_name}' to {prompt_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save prompt '{prompt_name}' to {prompt_file}: {str(e)}")
        return False

def delete_prompt(prompt_file: str, prompt_name: str) -> bool:
    try:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as file:
                prompts = json.load(file)
            
            prompts = [p for p in prompts if p['name'] != prompt_name]
            
            with open(prompt_file, 'w') as file:
                json.dump(prompts, file, indent=2)
            logger.info(f"Deleted prompt '{prompt_name}' from {prompt_file}")
            return True
        else:
            logger.warning(f"Prompt file {prompt_file} does not exist")
            return False
    except Exception as e:
        logger.error(f"Failed to delete prompt '{prompt_name}' from {prompt_file}: {str(e)}")
        return False

def get_available_prompts(prompt_files: List[str]) -> List[str]:
    all_prompts = set()
    for file in prompt_files:
        prompts = load_prompt_options([file])
        all_prompts.update(prompts.keys())
    return sorted(list(all_prompts))

def format_prompt(system_message: str, conversation_history: List[Dict[str, str]], prompt: str) -> str:
    formatted_system = format_system_message(system_message)
    formatted_history = format_conversation_history(conversation_history)
    formatted_prompt = format_user_prompt(prompt)
    return f"{formatted_system}{formatted_history}\n{formatted_prompt}"

def format_system_message(system_message: str) -> str:
    if system_message:
        return f"System: {system_message}\n\n"
    return ""

def format_conversation_history(history: List[Dict[str, str]]) -> str:
    formatted_history = ""
    for message in history:
        role = message['role'].capitalize()
        content = message['content']
        formatted_history += f"{role}: {content}\n"
    return formatted_history.strip() + "\n\n" if formatted_history else ""

def format_user_prompt(prompt: str) -> str:
    return f"Human: {prompt}\nAI:"

def format_full_prompt(system_message: str, conversation_history: List[Dict[str, str]], prompt: str) -> str:
    formatted_system = format_system_message(system_message)
    formatted_history = format_conversation_history(conversation_history)
    formatted_prompt = format_user_prompt(prompt)
    return f"{formatted_system}{formatted_history}\n{formatted_prompt}"