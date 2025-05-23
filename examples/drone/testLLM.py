import openai
import json
import base64
def openai_api_call(picture_path, json_inputs):
    """
    Uses OpenAI API to analyze the json_inputs using the specified prompt,
    then parses the output JSON into a Python data structure.

    :param picture_path: Path to the image file (ignored in this text-only version)
    :param json_inputs: Dictionary representing the current simulation state
    :return: Parsed response (assignments, waypoints, n_steps, reasoning)
    """

    # 1. Prepare prompts
    system_prompt = (
    """You are an expert multi-agent path planner for drone teams in a 3D environment. Your task is to assign drones to targets and generate the next collision-free waypoint for each drone, so that each drone hovers at a target point.
    Drones can be reassigned to different targets between planning steps.
    You must ensure drones’ paths and waypoints avoid collisions, keeping at least the minimum safe distance between all drones.
    Drones are considered to have reached their assigned target when their distance to it is less than the specified threshold and should then hover at that target.
    For each drone, output the absolute next waypoint it should move toward.
    You must also decide how many steps these waypoints are valid for (n_steps) before replanning is needed.
    Additionally, provide clear reasoning for your assignments and waypoint choices.
    Your output must be valid JSON, following the schema provided."""
    )

    user_prompt = (
        "Plan the next step for a team of drones to cover all targets in 3D space, following these rules:\n"
        "- Targets and drones may be reassigned at any step.\n"
        "- Drones should avoid colliding with each other.\n"
        f"- Minimum safe distance between drones: {json_inputs.get('min_distance', 0.1)} (meters)\n"
        f"- A drone is considered at its target if within {json_inputs.get('target_threshold', 0.1)} meters.\n"
        "- Provide only the next waypoint (absolute position) for each drone.\n"
        "- Suggest n_steps (the number of steps to follow these waypoints before the next planning round).\n"
        "- Output your reasoning.\n"
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
        "n_steps": 20,
        "reasoning": "Drone 0 is reassigned to target 1 to minimize total travel distance. The suggested waypoints keep drones at least 0.1 meters apart and move each drone closer to its target. The next waypoints are chosen to avoid head-on paths between drones 1 and 2. Drones should follow these waypoints for 20 steps before replanning, as this covers most of the remaining distance safely."
        }"""
    )

    # 2. Prepare messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    if picture_path:
        with open(picture_path, "rb") as img_file:
            img_bytes = img_file.read()
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        img_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{b64_img}"}
            ]
        }
        # For multimodal input, replace/add the user message accordingly
        messages = [messages[0], img_message]
    # 3. API call (assumes your OpenAI key is set up)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )

    # 4. Parse output: find the first JSON block
    output_text = response.choices[0].message.content.strip()
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

if __name__ == "__main__":
    # Example usage
    picture_path = None  # Replace with actual image path if needed
    json_inputs = {
    "step": 7,
    "targets": [
        [
        0.73,
        0.21,
        0.24
        ],
        [
        -0.15,
        0.41,
        0.04
        ],
        [
        -0.74,
        0.22,
        1.07
        ]
    ],
    "completed": [
        False,
        False,
        False
    ],
    "drone_states": [
        {
        "drone_id": 0,
        "step": 7,
        "target": [
            -0.15,
            0.41,
            0.04
        ],
        "position": [
            -0.2528819441795349,
            0.8843340277671814,
            0.3198320269584656
        ],
        "velocity": [
            0.0030767126008868217,
            -0.015517815947532654,
            -0.005785810295492411
        ],
        "attitude": [
            2.862884521484375,
            0.5816949009895325,
            -0.35064053535461426
        ],
        },
        {
        "drone_id": 1,
        "step": 7,
        "target": [
            0.73,
            0.21,
            0.24
        ],
        "position": [
            -0.07900696247816086,
            0.7555983066558838,
            0.31891781091690063
        ],
        "velocity": [
            0.03333578631281853,
            -0.01960943453013897,
            0.011130397208034992
        ],
        "attitude": [
            3.463578701019287,
            6.211928844451904,
            -2.0119924545288086
        ]
        },
        {
        "drone_id": 2,
        "step": 7,
        "target": [
            -0.74,
            0.22,
            1.07
        ],
        "position": [
            -0.03798650950193405,
            -0.21260909736156464,
            0.4382624924182892
        ],
        "velocity": [
            -0.04504018649458885,
            0.02746620588004589,
            0.04250222072005272
        ],
        "attitude": [
            -4.9467902183532715,
            -8.223052024841309,
            -0.4398237466812134
        ]
        
        }
    ]
    }
    response = openai_api_call(picture_path, json_inputs)
    print("Response:", response)