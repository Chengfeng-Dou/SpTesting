import os
import subprocess

from agent import ChatMemory
from doctor import Doctor
from patient import StanderPatient, TerminalPatient


class ChatRoom:
    def __init__(self, patient_proxy, doctor_proxy):
        self.patient_proxy = patient_proxy
        self.doctor_proxy = doctor_proxy

    def simulate(self):
        chat_memory: ChatMemory = self.patient_proxy.init_memory()
        print("模拟对话开始...")
        print("\033[92m", "患者:", chat_memory.history[0]["content"], "\033[0m")
        print()
        for i in range(5):
            doctor_response = self.doctor_proxy.chat([chat_memory])[0]
            print("\033[91m", "医生:", doctor_response, "\033[0m")
            print("-" * 80)
            if i < 4:
                patient_response = self.patient_proxy.chat(chat_memory, doctor_response)
            else:
                patient_response = "谢谢医生，再见！"
            print("\033[92m", "患者:", patient_response, "\033[0m")

            chat_memory.update("assistant", doctor_response)
            chat_memory.update("user", patient_response)
            if self.is_end(patient_response):
                break

            print()
        print("模拟对话结束！")
        return chat_memory

    def is_end(self, patient_response):
        return "对话结束" in patient_response


class TerminalDoctor:
    def chat(self, *args):
        response = input("医生:")
        return [response]


class GptDoctor:
    def __init__(self, model_path, lora_path=None, doctor_type="dcf"):
        self.model_path = model_path
        self.lora_path = lora_path
        print("Initializing doctor agent...")
        command = [
            "srun",
            "-p",
            "priv_jinzhi",
            "-M",
            "priv",
            "-N",
            "1",
            "--gres=gpu:1",
            "python",
            "doctor.py",
            "--model_path",
            self.model_path,
            "--doctor_type",
            doctor_type,
        ]

        if lora_path is not None:
            command.extend(["--lora_path", self.lora_path])

        self.proxy = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )

        for _ in range(10):
            output = self.proxy.stdout.readline()
            output = output.decode("utf-8")
            print(output.strip())
            if output.strip() == "have a try!":
                break
        print("The doctor agent is ready!")

    def chat(self, query: str):
        query = query.strip() + "\n"
        query = query.encode("utf-8")
        self.proxy.stdin.write(query)
        self.proxy.stdin.flush()
        response = self.proxy.stdout.readline().decode("utf-8")
        return response.strip()

    def clear_history(self):
        return self.chat("$$clear$$")

    def stop(self):
        return self.chat("$$stop$$")


def tp_gd():
    # 终端病人同 GPT 医生交谈
    doctor = Doctor("model/agent2")
    patient = TerminalPatient("你好，我叫张**，今年28岁。我最近在做牙齿矫正，戴牙套的时候出现了口腔溃疡的现象。今天情况加重，耳朵和牙龈都很痛。请问这种情况下该怎么办？")
    chat_room = ChatRoom(patient, doctor)
    memory = chat_room.simulate()
    print(doctor.stop())
    print(memory.pretty_history())


def sp_td():
    # 标准化病人同终端医生交谈
    patient = StanderPatient("neike/02_bronchial_asthma")
    doctor = TerminalDoctor()
    chat_room = ChatRoom(patient, doctor)
    chat_room.simulate()
    # print(memory.pretty_history())


def sp_gd():
    # 标准化病人同GPT之间进行测试
    patient = StanderPatient("neike/02_bronchial_asthma")
    doctor = Doctor("model/agent2")
    chat_room = ChatRoom(patient, doctor)
    chat_room.simulate()
    # print(doctor.clear_history())
    # print(doctor.stop())


def batch_test(doctor_type, sub_root, model):
    diseases = os.listdir(f"script/{sub_root}")
    print(diseases)
    doctor = Doctor(model)
    os.makedirs(f"history/{doctor_type}/{sub_root}", exist_ok=True)
    for disease in diseases:
        if disease.startswith("."):
            continue

        patient = StanderPatient(f"{sub_root}/{disease}")
        chat_room = ChatRoom(patient, doctor)
        chat_history = chat_room.simulate()
        chat_history.dump(f"history/{doctor_type}/{sub_root}/{disease}.jsonl")


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")
    batch_test("gpt4", "jingshenke", "model/gpt4")
