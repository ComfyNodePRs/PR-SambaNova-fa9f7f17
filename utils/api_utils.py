import requests
import json
import time
import logging
from typing import Dict, Any, Generator, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_api_request(data: Dict[str, Any], headers: Dict[str, str], url: str, max_retries: int) -> Tuple[Any, bool, str]:
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            logger.info(f"Response status: {response.status_code}")
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"Response body: {response.text}")
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if 'choices' in response_json and response_json['choices']:
                        return response_json, True, "200 OK"
                    else:
                        logger.warning("No valid response content found.")
                        return "No valid response content found.", False, "200 OK but no content"
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON response: {str(e)}")
                    return "Error parsing JSON response.", False, "200 OK but failed to parse JSON"
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded. Retrying after delay.")
                retry_after = int(response.headers.get('Retry-After', 5))
                time.sleep(retry_after)
            else:
                logger.error(f"Request failed with status code {response.status_code}")
                return response.text, False, f"{response.status_code} {response.reason}"
        except requests.RequestException as e:
            logger.error(f"Request failed on attempt {attempt + 1}: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error("Failed after all retries.")
    return "Failed after all retries.", False, "Failed after all retries"

def make_streaming_request(data: Dict[str, Any], headers: Dict[str, str], url: str) -> Generator[str, None, None]:
    try:
        with requests.post(url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith("data: "):
                            if line.strip() == "data: [DONE]":
                                logger.debug("Received end of stream")
                                break
                            try:
                                json_str = line[len("data: "):]
                                parsed_line = json.loads(json_str)
                                if 'choices' in parsed_line and parsed_line['choices']:
                                    content = parsed_line['choices'][0]['delta'].get('content', '')
                                    if content:
                                        yield content
                                else:
                                    logger.warning(f"Unexpected response format: {parsed_line}")
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse JSON: {json_str}")
                        else:
                            logger.warning(f"Unexpected line format: {line}")
                    else:
                        logger.debug("Received empty line in streaming response")
            else:
                error_message = f"Streaming request failed with status code {response.status_code}"
                logger.error(error_message)
                yield f"Error: {error_message}"
    except requests.RequestException as e:
        error_message = f"Streaming request failed: {str(e)}"
        logger.error(error_message)
        yield f"Error: {error_message}"

def validate_api_key(api_key: str, base_url: str) -> bool:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.get(f"{base_url}/chat/completions", headers=headers, timeout=10)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"API key validation failed. Status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        logger.error(f"API key validation request failed: {str(e)}")
        return False

def fine_tune_model(api_key: str, base_url: str, model: str, training_data: str, hyperparameters: Dict[str, Any]) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "training_file": training_data,
        "hyperparameters": hyperparameters
    }
    try:
        response = requests.post(f"{base_url}/fine-tunes", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            fine_tune_data = response.json()
            return fine_tune_data['id']
        else:
            logger.error(f"Failed to start fine-tuning. Status code: {response.status_code}")
            return ""
    except requests.RequestException as e:
        logger.error(f"Request to start fine-tuning failed: {str(e)}")
        return ""