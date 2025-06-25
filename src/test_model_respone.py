import os
import json
import pydra
from pydra import Config, REQUIRED
from multi_turn_config import MultiTurnConfig

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")

def read_turn_context(file_path: str, turn_id: int) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        turn_key = str(turn_id)
        if turn_key in json_data and "context" in json_data[turn_key]:
            print(f"Reading context for turn {turn_id} from {file_path}")
            return json_data[turn_key]["context"]
        else:
            print(f"Turn {turn_id} or context not found")
            return ""

    except FileNotFoundError:
        print(f"Log file not found: {file_path}")
        return ""
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return ""
    except Exception as e:
        print(f"Error reading log file: {e}")
        return ""


@pydra.main(base=MultiTurnConfig)
def main(config: MultiTurnConfig):
    config.set_preset_config(
        config.model_name if config.model_name is not None else "deepseek"
    )
    turn_id = 2
    log_dir = os.path.join(
        config.log_dir_prefix,
        config.run_group,
        config.run_name,
        "problem_" + str(config.problem_id),
        "sample_0",
    )
    file_path = os.path.join(log_dir, "log.json")

    context_content = read_turn_context(file_path, turn_id)
    print(f"Turn {turn_id} context: {context_content}")

    from openai import OpenAI

    client = OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        timeout=10000000,
        max_retries=3,
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "user",
                "content": context_content,
            },
        ],
        stream=True,
    )

    reasoning_content = ""
    content = ""
    print("[Reasoning]", end="", flush=True)
    for chunk in response:
        if chunk.choices[0].delta.reasoning_content:
            reasoning_chunk = chunk.choices[0].delta.reasoning_content
            reasoning_content += reasoning_chunk
            print(f"{reasoning_chunk}", end="", flush=True)
        elif chunk.choices[0].delta.content:
            content_chunk = chunk.choices[0].delta.content
            content += content_chunk
            print(content_chunk, end="", flush=True)

    print("\n\n--- FINAL RESPONSE ---")
    print(f"Reasoning: {reasoning_content}")
    print(f"Content: {content}")


if __name__ == "__main__":
    main()
