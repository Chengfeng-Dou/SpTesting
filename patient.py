from agent import ChatMemory, create_agent


class StanderPatient:
    def __init__(self, name):
        print(f"Initializing patient agent: {name}...")
        self.name = name
        self.agent = create_agent(name)

        with open(f"script/{name}/chief_complaint.txt") as fp:
            self.chief_complaint = "".join(fp.readlines()).replace(
                "请问该如何治疗？", ""
            )
        print("The patient agent is ready!")

    def init_memory(self):
        chat_memory = ChatMemory()
        chat_memory.update("user", self.chief_complaint)
        return chat_memory

    def chat(self, chat_memory: ChatMemory, query: str):
        history = chat_memory.pretty_history()
        inputs = {"chat_history": "\n".join(history), "query": query}
        response = self.agent(inputs)
        response = self.format_response(response)
        return response

    @staticmethod
    def format_response(response):
        result = response["result"]
        result.replace("：", ":")
        if '"' in result:
            if "content" in result:
                result = result.split(":", maxsplit=1)[-1]

            result = result.replace('"', "").strip()
        return result


class TerminalPatient:
    def __init__(self, chief_complaint):
        self.chief_complaint = chief_complaint

    def init_memory(self):
        chat_memory = ChatMemory()
        chat_memory.update("user", self.chief_complaint)
        return chat_memory

    def chat(self, *args):
        response = input("患者:")
        return response
