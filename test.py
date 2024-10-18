from openai import OpenAI

from modules.swarm import Agent, Swarm

# sys.path.append(os.path.join(os.getcwd(), ".."))


openai = OpenAI(
    api_key="tmp",
    base_url="http://192.168.100.128:4000",
)

swarm = Swarm(client=openai)


def get_current_time():
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


context_variables = {"current_time": get_current_time()}


# def instructions():
#     return """Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.
#     Current time is {current_time}.
#     """.format(
#         get_current_time()
#     )


triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
    parallel_tool_calls=True,
)


def transfer_back_to_triage():
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    return triage_agent


sthenno = Agent(
    name="sthenno",
    model="sthenno",
    instructions="You are Sthenno. 你扮演一位可爱的少女. 只使用中文回答. 和用户自然聊天. 你是用户的朋友. 你只回应闲聊相关内容.",
    functions=[transfer_back_to_triage],
    tool_choice="auto",
)


def transfer_to_sthenno():
    """Transfer to Sthenno. Sthenno is for common chatting, not for specific tasks."""
    return sthenno


triage_agent.functions.append(transfer_to_sthenno)


def query_wolfram(query: str):
    """Use WolframAlpha to solve any mathematical problem or get real-time informations.

    # WolframAlpha Instructions:
        - WolframAlpha understands natural language queries about entities in chemistry, physics, geography, history, art, astronomy, and more.
        - WolframAlpha performs mathematical calculations, date and unit conversions, formula solving, etc.
        - Convert inputs to simplified keyword queries whenever possible (e.g. convert "how many people live in France" to "France population").
        - Send queries in English only; translate non-English queries before sending, then respond in the original language.
        - Display image URLs with Markdown syntax: ![URL]
        - ALWAYS use this exponent notation: `6*10^14`, NEVER `6e14`.
        - ALWAYS use `"input": query` structure for queries to Wolfram endpoints; `query` must ONLY be a single-line string.
        - ALWAYS use proper Markdown formatting for all math, scientific, and chemical formulas, symbols, etc.:  '$$\n[expression]\n$$' for standalone cases and '\\( [expression] \\)' when inline.
        - Never mention your knowledge cutoff date; Wolfram may return more recent data.
        - Use ONLY single-letter variable names, with or without integer subscript (e.g., n, n1, n_1).
        - Use named physical constants (e.g., 'speed of light') without numerical substitution.
        - Include a space between compound units (e.g., "Ω m" for "ohm*meter").
        - To solve for a variable in an equation with units, consider solving a corresponding equation without units; exclude counting units (e.g., books), include genuine units (e.g., kg).
        - If data for multiple properties is needed, make separate calls for each property.
        - If a WolframAlpha result is not relevant to the query:
         -- If Wolfram provides multiple 'Assumptions' for a query, choose the more relevant one(s) without explaining the initial result. If you are unsure, ask the user to choose.
         -- Re-send the exact same 'input' with NO modifications, and add the 'assumption' parameter, formatted as a list, with the relevant values.
         -- ONLY simplify or rephrase the initial query if a more relevant 'Assumption' or other input suggestions are not provided.
         -- Do not explain each step unless user input is needed. Proceed directly to making a better API call based on the available assumptions.
    # After you receive the response from WolframAlpha:
        - Extract the relevant information from the response and send it back.
        - Return the response in a brief, clear, and concise manner, for example, \"The population of France is 67 million.\"
    """
    import requests

    url = "https://www.wolframalpha.com/api/v1/llm-api"
    params = {
        "input": query,
        "appid": "XLWH39-XHETWQ3K9E",
        "maxchars": 500,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Error querying WolframAlpha."


math_agent = Agent(
    name="Math Agent",
    model="gpt-4o-mini",
    instructions="Only answer math questions.",
    functions=[query_wolfram, transfer_back_to_triage],
)


def transfer_to_math_agent():
    """Transfer to Math Agent. Math Agent is for math questions."""
    return math_agent


triage_agent.functions.append(transfer_to_math_agent)


class Chat:
    def __init__(self):
        self.agent = triage_agent
        self.messages = []
        self.contents = []

    def update(self):
        response = swarm.run(agent=self.agent, messages=self.messages)
        print(f"response: {response}")

        self.contents = []
        for message in self.messages:
            if message["role"] != "assistant":
                continue
            if message.get("content"):
                print(f"message_got: {message["content"]}")
                self.contents.append(message["content"])
        self.messages.extend(response.messages)
        self.agent = response.agent

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        while self.contents == []:
            self.update()
        return self.contents[-1]


def chat(user_input):
    chat = Chat()
    while chat.chat(user_input) == []:
        pass
    return chat.chat(user_input)


if __name__ == "__main__":
    output = chat("What is the capital of France?")
    print(f"output: {output}")
