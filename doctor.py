from vllm import LLM, SamplingParams

from agent import ChatMemory


class Doctor:
    pwd = "/data0/dcf/rlhf"

    def __init__(self, model) -> None:
        self.model = LLM(
            f"{self.pwd}/{model}",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
        )

        self.config = SamplingParams(
            n=1,
            max_tokens=4096,
            temperature=0.,
            top_k=200,
            top_p=0.7,
            presence_penalty=1.1,
            stop=["</s>", "\n\n"],
        )

    def chat(self, chat_memories):
        inputs = [self.build_input(memory) for memory in chat_memories]
        outputs = self.model.generate(inputs, self.config, use_tqdm=False)

        ret = []
        for output in outputs:
            ret.append(output.outputs[0].text.strip())

        return ret

    def build_input(self, chat_memory: ChatMemory):
        ret = ["<reserved_108>多轮对话</s>"]
        for item in chat_memory.history:
            if item["role"] == "user":
                ret.append(f'<reserved_106>{item["content"]}</s>')
            else:
                ret.append(f'<reserved_107>{item["content"]}</s>')

        ret.append(f"<reserved_107>")
        return "".join(ret)
