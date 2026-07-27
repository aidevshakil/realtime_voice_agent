import datetime

import requests


def get_current_time(timezone: str = "local") -> str:
    now = datetime.datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_weather(city: str) -> str:
    try:
        resp = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException as exc:
        return f"Could not fetch weather for {city}: {exc}"


TOOLS = {
    "get_current_time": get_current_time,
    "get_weather": get_weather,
}

TOOL_DECLARATIONS = [
    {
        "name": "get_current_time",
        "description": "Get the current local date and time.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name to get weather for.",
                }
            },
            "required": ["city"],
        },
    },
]
