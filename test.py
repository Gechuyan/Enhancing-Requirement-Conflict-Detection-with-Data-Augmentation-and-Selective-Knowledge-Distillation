import logging

logging.getLogger("vllm").setLevel(logging.ERROR)

# from statsmodels.stats.inter_rater import fleiss_kappa
import csv
from huggingface_hub import InferenceClient
import os
import json
import re
import math
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import Accelerator
import torch
import sys
import openai
import google.generativeai as genai
from vllm import LLM, SamplingParams

# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from prompt_utils import detect_encoding
from prompt_utils import clean_content
from prompt_utils import get_response_nosteam, get_response_steam
from prompt_utils import JUDGE_TASK_PROMPT, JUDGE_EXAMPLE_PROMPT, JUDGE_SETTINGS_PROMPT
from prompt_utils import LLMS
from peft import PeftModel

MODEL_PATH = {
    "llama-3.2-3b-instruct": "./llama-3b-model",
    "llama-3.2-3b-instruct-post":"./output/llama_distill_llama-3.2-3b-instruct_0/final_merged_model",
    "qwen3-4b": "./qwen-4b-model",
    "qwen3-4b-post": "./output/qwen_distill_qwen3-4b_0/final_merged_model",
    "gemma-3-4b-it": "./gemma-4b-model",
    "gemma-3-4b-it-post": "./output/gemma_distill_gemma-3-4b-it_0/final_merged_model",
}
RESULT_ROOT_PATH = "./Test"
LOG_PATH = "test_log.txt"
TP_CONFIDENCE=[]
TN_CONFIDENCE=[]
FP_CONFIDENCE=[]
FN_CONFIDENCE=[]
CONFIDENCE=[]

class Judge:
    def __init__(self, use_api=False):
        # self.llms = ["qwen3-4b-post"]
        # self.llms = ["llama-3.2-3b-instruct-post"]
        self.llms = ["gemma-3-4b-it-post"]
        # self.llms=["qwen3-32b"]
        # self.llms=["deepseek-r1-distill-qwen-32b"]
        # self.llms=["gemma-3-27b-it"]
        # self.llms=["qwq-32b"]
        self.use_api = use_api
        if self.use_api == False:
            (
                self.llm,
                self.tokenizer,
            ) = self.get_local_model(MODEL_PATH[self.llms[0]])
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=512,
                stop_token_ids=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.pad_token_id,
                ],
            )

    def get_response(self, CONTENT, MODEL):
        # MODEL = MODEL.split("/")[-1].lower()
        # if "gemma" in MODEL or "llama" in MODEL:
        #     return self.get_modelscope_response(CONTENT,MODEL)
        # if "gemma" in MODEL or "llama" in MODEL:
        #     return self.get_huggingface_response(CONTENT,MODEL)
        # if "gemma" in MODEL :
        #     return self.get_gemma_response(CONTENT,MODEL)
        # if "llama" in MODEL :
        #     return self.get_huggingface_response(CONTENT,MODEL)
        system_content = (
            JUDGE_TASK_PROMPT + JUDGE_EXAMPLE_PROMPT + JUDGE_SETTINGS_PROMPT
        )
        response= get_response_steam(
            model=MODEL,
            user_content=CONTENT + "[REQ_END]",
            system_content=system_content,
            # enable_thinking=True,
            enable_thinking=False,
        )
        # content_list = response.choices[0].message.logprobs["content"]
        # target_confidence = 0.0

        # for item in content_list:
        #     token_str = item["token"].strip().lower()
        #     if "conflict" in token_str or "neutral" in token_str:
        #         logprob = item["logprob"]
        #         target_confidence = round(math.exp(logprob) * 100, 2)
        #         CONFIDENCE.append(target_confidence)
        #         break
        # return  response.choices[0].message.content
        return response

    def get_gemma_response(self, CONTENT, MODEL):
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
        system_content = (
            JUDGE_TASK_PROMPT + JUDGE_EXAMPLE_PROMPT + JUDGE_SETTINGS_PROMPT
        )
        user_content = CONTENT + "[REQ_END]"

        genai.configure(api_key="AAA")

        model = genai.GenerativeModel(model_name=MODEL)

        response = model.generate_content(system_content + "\n" + user_content)
        return response.text.strip()

    def get_huggingface_response(self, CONTENT, MODEL):
        system_content = (
            JUDGE_TASK_PROMPT + JUDGE_EXAMPLE_PROMPT + JUDGE_SETTINGS_PROMPT
        )
        user_content = CONTENT + "[REQ_END]"

        HF_TOKEN = "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"
        client = InferenceClient(model=MODEL, token=HF_TOKEN)

        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )

        return response.choices[0].message.content

    def get_modelscope_response(self, CONTENT, MODEL):
        system_content = (
            JUDGE_TASK_PROMPT + JUDGE_EXAMPLE_PROMPT + JUDGE_SETTINGS_PROMPT
        )
        user_content = CONTENT + "[REQ_END]"
        pipe = pipeline(
            task=Tasks.chat,
            model=MODEL,
            remote=True,  # 核心参数：不下载模型，用ModelScope云端资源
        )
        return pipe(
            {
                "text": system_content + "\n" + user_content,
                "history": [],  # 空对话历史，新手直接用
            }
        )

    def get_local_model(
        self, model_path, tensor_parallel_size=1, gpu_memory_utilization=0.8
    ):
        llm = LLM(
            model=model_path,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        if tokenizer.eos_token_id is None:
            if 'llama' in self.llms[0].lower():
                tokenizer.eos_token_id = 128009  # Llama3 默认常见 ID
            elif 'qwen' in self.llms[0].lower():
                tokenizer.eos_token_id = 151643  # qwen3 默认常见 ID
            elif 'gemma' in self.llms[0].lower():
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token) or tokenizer.convert_tokens_to_ids("<end_of_turn>") # gemma3 默认常见 ID
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return llm, tokenizer

    def get_local_response(self, CONTENT):
        return self.get_local_responses([CONTENT])[0]

    def get_local_responses(self, contents, tensor_parallel_size=1):
        prompts = []
        full_system_prompt = (
            JUDGE_TASK_PROMPT + JUDGE_EXAMPLE_PROMPT + JUDGE_SETTINGS_PROMPT
        )
        for content in contents:
            user_content = content + "[REQ_END]"
            messages = [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": user_content},
            ]
            text_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            prompts.append(text_prompt)

        outputs = self.llm.generate(prompts, self.sampling_params)
        responses = []
        for output in tqdm(outputs, desc="Adding responses", unit="output"):
            generated_text = output.outputs[0].text
            responses.append(generated_text)
        return responses

    def get_requirement_pairs(self, input_csv_path):
        requirement_pairs = []
        encoding = detect_encoding(input_csv_path)
        with open(input_csv_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=",")
            # next(reader)
            for row in reader:
                pair = tuple((row["Text1"], row["Text2"]))
                requirement_pairs.append(pair)
        return requirement_pairs

    def get_answers(self, input_csv_path):
        answers = []
        encoding = detect_encoding(input_csv_path)
        with open(input_csv_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=",")
            # next(reader)
            for row in reader:
                answers.append(row["Class"])
        return answers

    def test_llms(self, input_csv_path):
        accuracies = []
        precisions = []
        recalls = []
        f1s = []
        for index in range(len(self.llms)):
            llm = self.llms[index]
            data = (
                os.path.basename(input_csv_path)
                .replace("_clean_pairs", "")
                .replace(".csv", "")
            )
            llm = llm.split("/")[-1].split(":")[0].lower()
            result_path = f"{RESULT_ROOT_PATH}/{llm}/{data}/{llm}_{data}.csv"
            tp_path = result_path.replace(".csv", "_tp.csv")
            tn_path = result_path.replace(".csv", "_tn.csv")
            fp_path = result_path.replace(".csv", "_fp.csv")
            fn_path = result_path.replace(".csv", "_fn.csv")

            if (
                os.path.exists(tp_path)
                and os.path.exists(tn_path)
                and os.path.exists(fp_path)
                and os.path.exists(fn_path)
            ):
                accuracy, precision, recall, f1,macro_precision, macro_recall, macro_f1 = self.test_llm_cached(
                    input_csv_path, llm
                )

            else:
                requirement_pairs = self.get_requirement_pairs(input_csv_path)
                answers = self.get_answers(input_csv_path)
                all_judgments = self.get_all_judgments(
                    requirement_pairs, input_csv_path
                )
                llm_judgments = [judgments[index] for judgments in all_judgments]

                accuracy, precision, recall, f1,macro_precision, macro_recall, macro_f1 = self.test_llm(
                    input_csv_path,
                    llm,
                    requirement_pairs,
                    answers,
                    llm_judgments,
                )
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            accuracy_show = round(accuracy * 100, 2)
            precision_show = round(precision * 100, 2)
            recall_show = round(recall * 100, 2)
            f1_show = round(f1 * 100, 2)

            macro_precision_show=round(macro_precision*100,2)
            macro_recall_show=round(macro_recall*100,2)
            macro_f1_show=round(macro_f1*100,2)

            print(
                f"LLM = {self.llms[index]}: {accuracy_show = }%,{precision_show = }%,{recall_show = }%,{f1_show = }%\n{macro_precision_show = }%,{macro_recall_show = }%,{macro_f1_show = }%"
            )
            self.log(
                f"LLM = {self.llms[index]}: {accuracy_show = }%,{precision_show = }%,{recall_show = }%,{f1_show = }%\n{macro_precision_show = }%,{macro_recall_show = }%,{macro_f1_show = }%"
            )
        return accuracies, precisions, recalls, f1s

    def check_and_create(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Text1", "Text2", "Answer", "Judgment"])

    def test_llm_cached(self, input_csv_path, llm):
        tp = 0  # 真阳性：真实为1且预测为1
        tn = 0  # 真阴性：真实为0且预测为0
        fp = 0  # 假阳性：真实为0但预测为1
        fn = 0  # 假阴性：真实为1但预测为0

        data = (
            os.path.basename(input_csv_path)
            .replace("_clean_pairs", "")
            .replace(".csv", "")
        )
        llm = llm.split("/")[-1].split(":")[0].lower()
        result_path = f"{RESULT_ROOT_PATH}/{llm}/{data}/{llm}_{data}.csv"
        tp_path = result_path.replace(".csv", "_tp.csv")
        tn_path = result_path.replace(".csv", "_tn.csv")
        fp_path = result_path.replace(".csv", "_fp.csv")
        fn_path = result_path.replace(".csv", "_fn.csv")

        with open(tp_path, "r", encoding=detect_encoding(tp_path)) as f:
            tp = sum(1 for _ in f) - 1
        with open(tn_path, "r", encoding=detect_encoding(tn_path)) as f:
            tn = sum(1 for _ in f) - 1
        with open(fp_path, "r", encoding=detect_encoding(fp_path)) as f:
            fp = sum(1 for _ in f) - 1
        with open(fn_path, "r", encoding=detect_encoding(fn_path)) as f:
            fn = sum(1 for _ in f) - 1

        print(f"{tp = },{tn = },{fp = },{fn = }")
        self.log(f"{tp = },{tn = },{fp = },{fn = }")
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_n = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_n = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_n = 2 * (precision_n * recall_n) / (precision_n + recall_n) if (precision_n + recall_n) > 0 else 0

        macro_precision = (precision + precision_n) / 2
        macro_recall = (recall + recall_n) / 2
        macro_f1 = (f1 + f1_n) / 2

        return accuracy, precision, recall, f1,macro_precision, macro_recall, macro_f1

    def test_llm(self, input_csv_path, llm, requirement_pairs, answers, judgments):
        tp = 0  # 真阳性：真实为1且预测为1
        tn = 0  # 真阴性：真实为0且预测为0
        fp = 0  # 假阳性：真实为0但预测为1
        fn = 0  # 假阴性：真实为1但预测为0

        data = (
            os.path.basename(input_csv_path)
            .replace("_clean_pairs", "")
            .replace(".csv", "")
        )
        llm = llm.split("/")[-1].split(":")[0].lower()
        result_path = f"{RESULT_ROOT_PATH}/{llm}/{data}/{llm}_{data}.csv"
        tp_path = result_path.replace(".csv", "_tp.csv")
        tn_path = result_path.replace(".csv", "_tn.csv")
        fp_path = result_path.replace(".csv", "_fp.csv")
        fn_path = result_path.replace(".csv", "_fn.csv")

        self.check_and_create(tp_path)
        self.check_and_create(tn_path)
        self.check_and_create(fp_path)
        self.check_and_create(fn_path)

        for index, answer in enumerate(answers):
            judgment = judgments[index]
            req0 = clean_content(requirement_pairs[index][0])
            req1 = clean_content(requirement_pairs[index][1])

            if not self.is_conflict(answer) and not self.is_conflict(judgment):
                tn += 1
                with open(tn_path, "a", encoding=detect_encoding(tn_path)) as f:
                    f.write(f'{index},"{req0}","{req1}","{answer}","{judgment}"\n')
                # TN_CONFIDENCE.append(CONFIDENCE[index])
            elif not self.is_conflict(answer) and self.is_conflict(judgment):
                fp += 1
                with open(fp_path, "a", encoding=detect_encoding(fp_path)) as f:
                    f.write(f'{index},"{req0}","{req1}","{answer}","{judgment}"\n')
                # FP_CONFIDENCE.append(CONFIDENCE[index])
            elif self.is_conflict(answer) and not self.is_conflict(judgment):
                fn += 1
                with open(fn_path, "a", encoding=detect_encoding(fn_path)) as f:
                    f.write(f'{index},"{req0}","{req1}","{answer}","{judgment}"\n')
                # FN_CONFIDENCE.append(CONFIDENCE[index])
            elif self.is_conflict(answer) and self.is_conflict(judgment):
                tp += 1
                with open(tp_path, "a", encoding=detect_encoding(tp_path)) as f:
                    f.write(f'{index},"{req0}","{req1}","{answer}","{judgment}"\n')
                # TP_CONFIDENCE.append(CONFIDENCE[index])
            else:
                print(f"{judgment.lower()},{answer.lower()}")

        print(f"{tp = },{tn = },{fp = },{fn = }")
        self.log(f"{tp = },{tn = },{fp = },{fn = }")
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precision_n = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_n = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_n = 2 * (precision_n * recall_n) / (precision_n + recall_n) if (precision_n + recall_n) > 0 else 0

        macro_precision = (precision + precision_n) / 2
        macro_recall = (recall + recall_n) / 2
        macro_f1 = (f1 + f1_n) / 2

        return accuracy, precision, recall, f1, macro_precision, macro_recall, macro_f1

    def is_conflict(self, answer):
        answer = answer.lower()
        return not (answer == "neutral" or answer == "0")

    def write_cache(self, index, requirement_pair, judgment, reason, result_path):
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        requirement_pair = list(requirement_pair)
        requirement_pair[0] = clean_content(requirement_pair[0])
        requirement_pair[1] = clean_content(requirement_pair[1])
        judgment = clean_content(judgment)
        reason = clean_content(reason)

        if os.path.exists(result_path):
            encoding = detect_encoding(result_path)
            with open(result_path, "a", encoding=encoding) as f:
                f.write(
                    f'{index},"{requirement_pair[0]}","{requirement_pair[1]}","{judgment}","{reason}"\n'
                )
        else:
            with open(result_path, "w", encoding="utf-8") as f:
                f.write("Id,Text1,Text2,Class,Reason\n")
                f.write(
                    f'{index},"{requirement_pair[0]}","{requirement_pair[1]}","{judgment}","{reason}"\n'
                )

    def get_last_id(self, result_path):
        if os.path.exists(result_path):
            encoding = detect_encoding(result_path)
            with open(result_path, "r", encoding=encoding) as f:
                reader = csv.DictReader(f, delimiter=",")
                last_id = list(reader)[-1]["Id"]
            return last_id
        return -1

    def get_cache(self, index, result_path):
        encoding = detect_encoding(result_path)
        # print(result_path)
        with open(result_path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=",")
            reader = list(reader)
            if reader[index]["Id"] == str(index):
                # if reader[index] is not None:
                return reader[index]["Class"]
            else:
                print(index)
                sys.exit(1)
                # return "Not Cached"

    def get_all_judgments(self, requirement_pairs, input_csv_path):
        all_judgments = []
        result_paths = []
        for llm in self.llms:
            llm = llm.split("/")[-1].split(":")[0].lower()
            data = (
                os.path.basename(input_csv_path)
                .replace("_clean_pairs", "")
                .replace(".csv", "")
            )
            result_path = f"{RESULT_ROOT_PATH}/{llm}/{data}/{llm}_{data}.csv"
            result_paths.append(result_path)
        last_ids = []
        for result_path in result_paths:
            last_id = self.get_last_id(result_path)
            last_ids.append(last_id)
        if self.use_api == True:
            for pair_index, requirement_pair in enumerate(
                tqdm(requirement_pairs, desc="Juding requirements")
            ):
                judgments = []
                for llm_index, llm in enumerate(self.llms):
                    # print(f"{last_ids = }")
                    if pair_index <= int(last_ids[llm_index]):
                        judgments.append(
                            self.get_cache(pair_index, result_paths[llm_index])
                        )
                    else:
                        model = llm
                        content = (
                            "Req1: "
                            + requirement_pair[0]
                            + "\n"
                            + "Req2: "
                            + requirement_pair[1]
                        )
                        for _ in range(5):
                            # API
                            response = self.get_response(
                                CONTENT=content,
                                MODEL=model,
                            )
                            judgment, success = self.wash_response(response)
                            if success:
                                break
                        reason = ""
                        self.write_cache(
                            index=pair_index,
                            requirement_pair=requirement_pair,
                            judgment=judgment,
                            reason=reason,
                            result_path=result_paths[llm_index],
                        )
                        judgments.append(judgment)
                all_judgments.append(judgments)
            return all_judgments
        else:
            contents = []
            for requirement_pair in requirement_pairs:
                content = (
                    "Req1: "
                    + requirement_pair[0]
                    + "\n"
                    + "Req2: "
                    + requirement_pair[1]
                )
                contents.append(content)
            responses = self.get_local_responses(
                contents=contents, tensor_parallel_size=1
            )
            for pair_index, response in enumerate(responses):
                judgments = []
                for llm_index, llm in enumerate(self.llms):
                    judgment, success = self.wash_response(response)
                    if not success:
                        for _ in range(5):
                            response = self.get_local_response(
                                CONTENT=contents[pair_index],
                            )
                            judgment, success = self.wash_response(response)
                            if success:
                                break
                    reason = ""
                    self.write_cache(
                        index=pair_index,
                        requirement_pair=requirement_pairs[pair_index],
                        judgment=judgment,
                        reason=reason,
                        result_path=result_paths[llm_index],
                    )
                    judgments.append(judgment)
                all_judgments.append(judgments)
            return all_judgments
        # return []

    def log(self, content=""):
        dir_path = os.path.dirname(LOG_PATH)
        if dir_path and not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH, exist_ok=True)
        try:
            with open(LOG_PATH, "a+", encoding="utf-8") as f:
                f.write(content + "\n")
            # print(f"✅ 内容已成功写入文件: {LOG_PATH}")
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

    def wash_response(self, response):
        response = response.replace("```json", "").replace("```", "").strip()
        try:
            tag_pattern = r'"Tag"\s*:\s*"([^"]+)"'
            match = re.search(tag_pattern, response, re.IGNORECASE)
            if match:
                raw_tag = match.group(1).strip()
                judgment = raw_tag.lower()
                if judgment in ["neutral", "conflict"]:
                    return judgment, True
                else:
                    raise ValueError("judgment is not conflict or neutral")
            else:
                if "conflict" in response.lower():
                    return "conflict", True
                elif "neutral" in response.lower():
                    return "neutral", True
                else:
                    return "", False
        except Exception as e:
            print(f"{type(e).__name__} | {e}")
            print(response)
            return "", False
        
    def record_confidence(self):
        tp_path=f"./tp_confidence_{self.llms[0].split('/')[-1].split(':')[0].lower()}.txt"
        tn_path=f"./tn_confidence_{self.llms[0].split('/')[-1].split(':')[0].lower()}.txt"
        fp_path=f"./fp_confidence_{self.llms[0].split('/')[-1].split(':')[0].lower()}.txt"
        fn_path=f"./fn_confidence_{self.llms[0].split('/')[-1].split(':')[0].lower()}.txt"
        with open(tp_path,"a+",encoding="utf-8") as f:
            for item in TP_CONFIDENCE:
                f.write(f"{item}\n")
        with open(tn_path,"a+",encoding="utf-8") as f:
            for item in TN_CONFIDENCE:
                f.write(f"{item}\n")
        with open(fp_path,"a+",encoding="utf-8") as f:
            for item in FP_CONFIDENCE:
                f.write(f"{item}\n")
        with open(fn_path,"a+",encoding="utf-8") as f:
            for item in FN_CONFIDENCE:
                f.write(f"{item}\n")

    def main(self, test_csv_path, train_csv_path=""):
        self.test_llms(test_csv_path)
        # self.record_confidence()


if __name__ == "__main__":
    judge = Judge(use_api=False)

    test_csv_paths = [
        "Dataset/test/alice.csv",
        "Dataset/test/cn.csv",
        "Dataset/test/ibmuav.csv",
        "Dataset/test/opencoss.csv",
        "Dataset/test/pure.csv",
        "Dataset/test/uav.csv",
        "Dataset/test/worldvista.csv",
    ]

    LOG_PATH = LOG_PATH.replace(
        ".txt", "_" + judge.llms[0].split("/")[-1].split(":")[0] + ".txt"
    ).lower()
    if os.path.isfile(LOG_PATH):
        os.remove(LOG_PATH)

    for i in range(len(test_csv_paths)):
        test_csv_path = test_csv_paths[i]
        print(f"{test_csv_path},{judge.llms[0]}")
        judge.log(f"{test_csv_path},{judge.llms[0]}")
        judge.main(test_csv_path)
