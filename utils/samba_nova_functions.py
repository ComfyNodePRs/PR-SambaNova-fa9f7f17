import json
import logging
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_functions() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of available functions that can be called by the language model.
    """
    return {
        "get_current_weather": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
        "get_stock_price": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The stock symbol, e.g. AAPL for Apple Inc.",
                    },
                },
                "required": ["symbol"],
            },
        },
    }

def call_function(function_name: str, arguments: Dict[str, Any]) -> str:
    """
    Calls the specified function with the given arguments.
    """
    available_functions = get_available_functions()
    if function_name not in available_functions:
        return f"Error: Function '{function_name}' not found."

    try:
        if function_name == "get_current_weather":
            return get_current_weather(arguments.get("location"), arguments.get("unit", "celsius"))
        elif function_name == "get_stock_price":
            return get_stock_price(arguments.get("symbol"))
        else:
            return f"Error: Function '{function_name}' is not implemented."
    except Exception as e:
        logger.error(f"Error calling function '{function_name}': {str(e)}")
        return f"Error: Failed to call function '{function_name}'."

def get_current_weather(location: str, unit: str = "celsius") -> str:
    """
    Simulates getting the current weather for a given location.
    In a real implementation, this would make an API call to a weather service.
    """
    # This is a mock implementation
    weather_data = {
        "San Francisco, CA": {"temperature": 18, "condition": "Foggy"},
        "New York, NY": {"temperature": 22, "condition": "Sunny"},
        "London, UK": {"temperature": 15, "condition": "Rainy"},
    }

    if location not in weather_data:
        return f"Weather data not available for {location}"

    temp = weather_data[location]["temperature"]
    if unit == "fahrenheit":
        temp = (temp * 9/5) + 32

    return f"The current weather in {location} is {temp}°{'C' if unit == 'celsius' else 'F'} and {weather_data[location]['condition']}."

def get_stock_price(symbol: str) -> str:
    """
    Simulates getting the current stock price for a given symbol.
    In a real implementation, this would make an API call to a stock price service.
    """
    # This is a mock implementation
    stock_data = {
        "AAPL": 150.25,
        "GOOGL": 2800.75,
        "MSFT": 300.50,
    }

    if symbol not in stock_data:
        return f"Stock price not available for symbol {symbol}"

    return f"The current stock price of {symbol} is ${stock_data[symbol]:.2f}"

def parse_function_call(function_call: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parses the function call object returned by the language model.
    """
    if not isinstance(function_call, dict) or 'name' not in function_call or 'arguments' not in function_call:
        logger.error("Invalid function call object")
        return None

    try:
        arguments = json.loads(function_call['arguments'])
    except json.JSONDecodeError:
        logger.error("Failed to parse function arguments")
        return None

    return {
        'name': function_call['name'],
        'arguments': arguments
    }

def execute_function_call(function_call: Dict[str, Any]) -> str:
    """
    Executes a function call parsed from the language model's response.
    """
    parsed_call = parse_function_call(function_call)
    if parsed_call is None:
        return "Error: Failed to parse function call"

    return call_function(parsed_call['name'], parsed_call['arguments'])