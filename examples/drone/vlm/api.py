from openai import OpenAI
import json
import base64
import os
import time
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


def openai_api_call(picture_path, json_inputs, model_name="gpt-4o", conversation_history=None, interval=200):
    """
    Uses OpenAI API to analyze the json_inputs using the specified prompt,
    with conversation history support.

    :param picture_path: Path to the image file (None for text-only mode)
    :param json_inputs: Dictionary representing the current simulation state
    :param model_name: Name of the model to use
    :param conversation_history: List of previous interactions for context
    :return: Parsed response (assignments, waypoints, n_steps, reasoning)
    """
    # 1. Get API key
    if model_name in ["gpt-4o", "o4-mini"]:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI(api_key=openai_api_key)
    elif model_name == "gemini-2.5-pro":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        client = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    # 2. Prepare system prompt
    system_prompt = (
    """You are an expert multi-agent path planner for drone teams in a 3D environment. Your task is to assign drones to targets and generate the next collision-free waypoint for each drone, so that each drone hovers at a target point.

Key Requirements:
- Drones can be reassigned to different targets between planning steps.
- You must ensure drones' paths and waypoints avoid collisions, keeping at least the minimum safe distance between all drones.
- Drones are considered to have reached their assigned target when their distance to it is less than the specified threshold.
- IMPORTANT: Always provide waypoints for ALL drones, even if they have reached their targets.
- For drones that have completed their targets: set their waypoint to their assigned target position (they should hover at the target).
- For drones still moving to targets: provide the next waypoint they should move toward.
- You must also decide how many steps these waypoints are valid for (n_steps) before replanning is needed.
- Additionally, provide clear reasoning for your assignments and waypoint choices.
- Your output must be valid JSON, following the schema provided.
- Learn from previous interactions to improve your planning decisions."""
    )

    # 3. Prepare collision information
    collision_info = ""
    if json_inputs.get('collision_detected', False):
        collision_info = f"\n🚨 URGENT: COLLISION RISK DETECTED! 🚨\n"
        collision_info += f"The following drone pairs are within the minimum safe distance:\n"
        for pair in json_inputs.get('collision_pairs', []):
            collision_info += f"- Drones {pair['drone_1']} and {pair['drone_2']} are {pair['distance']:.3f}m apart (min: {json_inputs.get('min_distance', 0.1)}m)\n"
        collision_info += f"IMMEDIATE collision avoidance is required! Prioritize separating these drones safely.\n"
    
    # 4. Prepare current user prompt without history
    user_prompt = (
        "Plan the next step for a team of drones to cover all targets in 3D space, following these rules:\n"
        "- Targets and drones may be reassigned at any step.\n"
        "- Drones should avoid colliding with each other.\n"
        f"- Minimum safe distance between drones: {json_inputs.get('min_distance', 0.1)} (meters)\n"
        f"- A drone is considered at its target if within {json_inputs.get('target_threshold', 0.1)} meters.\n"
        "- Provide waypoints for ALL drones (including completed ones - set to target position).\n"
        f"- Suggest n_steps (the number of steps to follow these waypoints before the next planning round) suggested by the previous interactions. A default value of {interval} is suggested if no previous interactions are available.\n"
        "- Output your reasoning.\n"
        f"{collision_info}"
        "Current State (JSON):\n"
        f"{json.dumps(json_inputs, indent=2)}\n"
        "Your output should follow this JSON schema:\n"
        "{\n"
        '  "assignments": [\n'
        '    {"drone_id": int, "target_id": int}\n'
        '  ],\n'
        '  "waypoints": [\n'
        '    {"drone_id": int, "waypoint": [x, y, z]}\n'
        '  ],\n'
        '  "n_steps": int,\n'
        '  "reasoning": "string"\n'
        "}\n"
        """Sample Output:
        {
        "assignments": [
            {"drone_id": 0, "target_id": 1},
            {"drone_id": 1, "target_id": 0},
            {"drone_id": 2, "target_id": 2}
        ],
        "waypoints": [
            {"drone_id": 0, "waypoint": [0.1, 0.5, 0.1]},
            {"drone_id": 1, "waypoint": [0.7, 0.2, 0.2]},
            {"drone_id": 2, "waypoint": [-0.7, 0.2, 1.0]}
        ],
        "n_steps": 200,
        "reasoning": "Drone 0 is reassigned to target 1 to minimize total travel distance. The suggested waypoints keep drones at least 0.1 meters apart and move each drone closer to its target. Drone 2 has reached its target so its waypoint is set to the target position for hovering."
        }"""
    )

    # 5. Build messages with conversation history as structured messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history as structured messages
    if conversation_history and len(conversation_history) > 0:
        for interaction in conversation_history[-3:]:  # Use last 3 interactions
            # Add user message for the previous input
            hist_user_content = f"Previous planning request (Step {interaction['step']}):\n{json.dumps(interaction['input'], indent=2)}"
            messages.append({"role": "user", "content": hist_user_content})
            
            # Add assistant response if it was successful
            if interaction['response']:
                hist_assistant_content = json.dumps(interaction['response'], indent=2)
                messages.append({"role": "assistant", "content": hist_assistant_content})
            else:
                # If there was no response, add a note about the failure
                messages.append({"role": "assistant", "content": "I was unable to provide a valid response for this request."})
    
    # Add current user message
    if picture_path:
        # For vision-enabled requests
        with open(picture_path, "rb") as img_file:
            img_bytes = img_file.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        current_user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
            ]
        }
    else:
        # For text-only requests
        current_user_message = {"role": "user", "content": user_prompt}
    
    messages.append(current_user_message)

    # 6. API call 
    max_retries = 3
    retry_delay = 2
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API call failed, retrying in {retry_delay} seconds... ({attempt+1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"API call failed after {max_retries} attempts: {str(e)}")
                return None

    # 7. Parse output: find the first JSON block
    output_text = completion.choices[0].message.content.strip()
    try:
        # Find the first '{' and last '}' (most robust for JSON)
        start = output_text.find('{')
        end = output_text.rfind('}') + 1
        if start == -1 or end == -1:
            raise ValueError("No JSON object found in model output.")
        json_part = output_text[start:end]
        parsed = json.loads(json_part)
    except Exception as e:
        print("Failed to parse GPT output:", e)
        print("Raw output:\n", output_text)
        return None

    return parsed