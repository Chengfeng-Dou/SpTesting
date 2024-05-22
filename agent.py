from collections import defaultdict
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
)
from typing import Any, Dict, Optional
import inspect
import json
import jsonlines

from tqdm import tqdm

from retriever import read_vector_store
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from retriever import read_vector_store
import config

from langchain.schema import HumanMessage, SystemMessage


class StanderPatientChain(RetrievalQA):
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        chat_history = inputs["chat_history"]
        question = inputs[self.input_key]

        if len(chat_history) > 0:
            query = chat_history.split("\n")[-4:]
            query = "\n".join(query) + "\n" + question
        else:
            query = question

        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(query, run_manager=_run_manager)
        else:
            docs = self._get_docs(query)  # type: ignore[call-arg]
        answer = self.combine_documents_chain.run(
            input_documents=docs,
            question=question,
            chat_history=chat_history,
            callbacks=_run_manager.get_child(),
        )

        if self.return_source_documents:
            return {self.output_key: answer, "source_documents": docs}
        else:
            return {self.output_key: answer}


def create_agent(name):
    vector_store = read_vector_store(name)
    template = "\n".join(
        [
            "请扮演一个患者，同医生交流。需要满足下述要求：",
            "1. 如果医生提问，请根据知识库当中的内容以及对话历史的记录进行回答，不要回复超出提问的内容。",
            "2. 如果医生推荐你去做检查，请告诉他检查结果，不要只说自己做了该项检查，也不要随便编造检查结果，请根据知识库回答。",
            "3. 除非医生主动提问，否则不要暴露自己的任何信息给医生，请被动的接受医生引导。",
            "4. 如果医生没有提问，可以询问医生自己得了什么病以及该如何治疗。"
            "知识库：{context}",
            "对话历史：{chat_history}",
            "医生： {question}",
            "你的回复：",
        ]
    )

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)
    agent = StanderPatientChain.from_chain_type(
        ChatOpenAI(
            model_name=config.MODEL,
            temperature=0.5,
            openai_api_base=config.OPEN_AI_BASE,
            openai_api_key=config.OPEN_AI_KEY,
            max_retries=10,
        ),
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    return agent


class ChatMemory:
    def __init__(self) -> None:
        self.history = []

    def update(self, role, content):
        self.history.append({"role": role, "content": content})

    def pretty_history(self):
        history_ = []
        for item in self.history:
            role = "患者" if item["role"] == "user" else "医生"
            history_.append(f"{role}：{item['content']}")

        return history_

    def dump(self, filepath):
        with jsonlines.open(filepath, "w") as writer:
            writer.write_all(self.history)

    def load(self, filepath):
        with jsonlines.open(filepath, "r") as reader:
            for line in reader:
                self.history.append(line)


class EvaluationAgent:
    def __init__(self, name) -> None:
        self.check_list = json.load(open(f"check_list/{name}.json"))
        self.chat_model = ChatOpenAI(
            model_name=config.MODEL,
            temperature=0,
            openai_api_base=config.OPEN_AI_BASE,
            openai_api_key=config.OPEN_AI_KEY,
        )

    def evaluate(self, chat_memory: ChatMemory):
        history = "\n".join(chat_memory.pretty_history())
        history = f"<对话>\n{history}\n</对话>"
        prompts = defaultdict(list)

        for item in self.check_list["consultation_content"]:
            prompts["重点问诊考核"].append(
                f"{history}\n请问在上述对话中，医生{item}\nA. 是\nB. 否"
            )

        for item in self.check_list["diagnostic"]:
            prompts["疾病诊断考核"].append(
                f"{history}\n请问在上述对话中，医生是否提及患者可能患有{item}?\nA. 是\n B. 否"
            )

        for item in self.check_list["medical_checkup"]:
            prompts["医学检查考核"].append(
                f"{history}\n请问在上述对话中，医生是否建议患者去做{item}?\nA. 是\nB. 否"
            )

        results = {}
        for key, value in tqdm(prompts.items(), total=3):
            score = 0.0
            messages = [
                [
                    SystemMessage(
                        content="请阅读下列对话，完成选择题。要求首先生成对题目的分析，然后返回选择结果。"
                    ),
                    HumanMessage(content=v),
                ]
                for v in value
            ]

            result = self.chat_model.generate(messages)
            details = []
            for q, a in zip(value, result.generations):
                q = q.split("\n")[-3]
                a = a[0].text
                if "A" in a:
                    score += 1

                details.append({"question": q, "answer": a})

            results[key] = {"score": score / len(value), "details": details}

        return results


if __name__ == "__main__":
    m = ChatMemory()
    m.load("diag_history/dpo/01_01_v3.json")
    eval_agent = EvaluationAgent("01_severe_pneumonia")
    r = eval_agent.evaluate(m)
    with open("result/dpo/01_01_v3.json", "w") as fp:
        json.dump(r, fp, ensure_ascii=False)
