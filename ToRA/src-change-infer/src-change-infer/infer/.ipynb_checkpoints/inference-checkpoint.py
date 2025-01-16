"""
This script support vllm batch inference with cot/pal/tora prompt.
Also sopport inference of fine-tuned models like WizardMath/ToRA.
Code based on: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""
import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm

from eval.evaluate import evaluate
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="gsm8k", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tora", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int) # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=1024, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_train_prompt_format", action="store_true")
    args = parser.parse_args()
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy sampling (vllm)
    return args

def flatten_conditions(condition_lists):
    """
    将条件列表扁平化，按行拆分并去除空白行。
    """
    flattened = []  # 用于存储处理后的条件

    for condition in condition_lists:
        # 按行拆分条件，去掉每行的前后空白字符
        lines = condition.splitlines()
        # 处理每行，去掉前后空格
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:  # 如果去掉空白后行不为空，则加入结果列表
                flattened.append(stripped_line)

    return flattened


def prepare_data(args):
    examples = load_data(args.data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        examples = random.sample(examples, args.num_test_sample)
    elif args.num_test_sample == -1:
        args.num_test_sample = len(examples)
    
    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    if args.end == -1:
        args.end = len(examples)
    examples = examples[args.start:args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_s{args.start}_e{args.end}_{dt_string}.jsonl'
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)

    # load all processed samples
    processed_files = [f for f in os.listdir(f"{args.output_dir}/{model_name}/{args.data_name}/") if f.endswith(".jsonl") and f.startswith(out_file_prefix)]    
    processed_samples = []
    for f in processed_files:
        processed_samples.extend(list(load_jsonl(f"{args.output_dir}/{model_name}/{args.data_name}/{f}")))

    # dedepulicate
    processed_samples = {sample['idx']: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    total_examples = len(examples)
    examples = [example for example in examples if example['idx'] not in processed_idxs]
    print(f"Idx {args.start} - {args.end}: Remain {len(examples)}/{total_examples} samples.")
    if len(examples) == 0:
        pass
    else:
        print(examples[0])
    return examples, processed_samples, out_file


def main(args):
    examples, processed_samples, out_file = prepare_data(args)

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr='solution()')
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    # load model
    if len(examples) > 0:
        available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=len(available_gpus))
    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example['idx']

        # parse question and answer
        example['question'] = parse_question(example, args.data_name)
        gt_cot, gt_ans = parse_ground_truth(example, args.data_name)
        full_prompt = construct_prompt(args, example)

        sample = {'idx': idx, 'question': example['question'], 'gt_cot': gt_cot, 'gt': gt_ans, 'prompt': full_prompt}

        # add remain fields
        for key in ['level', 'type', 'unit', 'solution_type', 'choices', 'solution', 'ques_type', \
            'ans_type', 'answer_type', 'dataset', 'subfield', 'filed', 'theorem', 'answer']:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)  

    print("dataset:", args.data_name, "samples:", len(samples))
    if len(samples) > 0:
        print("-" * 50)
        print("sample:", samples[0]['prompt'])
        print("-" * 50)

    # repeat n times
    remain_prompts = [sample['prompt'] for sample in samples for _ in range(args.n_sampling)]
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    max_func_call = 1 if args.prompt_type in ['cot', 'pal'] else 4
    stop_tokens = ["</s>", "```output"]

    if args.prompt_type in ['cot']:
        stop_tokens.append("\n\n")
    elif args.prompt_type in ['wizard_zs', 'platypus_fs']:
        stop_tokens.extend(["Instruction", "Response"])

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("=" * 50, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        if epoch == 0:
            # Step 1: 重构问题
            original_prompts = [item[1] for item in current_prompts]
            reconstructed_prompts = [
                f"Given the original problem: \"{original}\", please give the concrete prompt (problem) that can generate this answer. "
                f"The problem should contain all basic and necessary information and correspond to the answer. "
                f"The problem can only ask for one result."
                for original in original_prompts
            ]

            # Step 2: 利用 LLM 确定重构问题
            reconstructed_outputs = llm.generate(reconstructed_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_tokens,
            ))

            # 提取重构后的问题文本
            reconstructed_questions = [
                output.outputs[0].text for output in reconstructed_outputs
            ]

            # Step 3: 分解原问题和重构问题
            # 初始化条件列表
            original_conditions = []
            reconstructed_conditions = []

            # 准备用于批量处理的提示
            orig_cond_prompts = []
            recon_cond_prompts = []

            orig_cond_prompt = (
                "Please list the conditions of the problem. There may be multiple conditions.\n"
                "Do not list conditions not related to calculations, but list all necessary conditions.\n"
                "The format should be:\n"
                "Conditions:\n"
                "This is your output of conditions. Each line is one condition."
            )

            # 创建提示用于原问题和重构问题
            for original, reconstructed in zip(original_prompts, reconstructed_questions):
                orig_cond_prompts.append(original + orig_cond_prompt)  # 添加原问题的提示
                recon_cond_prompts.append(reconstructed + orig_cond_prompt)  # 添加重构问题的提示

            # 批量处理原问题的条件提取
            orig_cond_outputs = llm.generate(orig_cond_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,  # 批量生成数量
                stop=stop_tokens,
            ))

            # 提取原条件
            for output in orig_cond_outputs:
                original_conditions.append(output.outputs[0].text.splitlines())

            # 批量处理重构问题的条件提取
            recon_cond_outputs = llm.generate(recon_cond_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,  # 批量生成数量
                stop=stop_tokens,
            ))

            # 提取重构条件
            for output in recon_cond_outputs:
                reconstructed_conditions.append(output.outputs[0].text.splitlines())

            flattened_original_conditions = [flatten_conditions(conditions) for conditions in original_conditions]
            flattened_reconstructed_conditions = [flatten_conditions(conditions) for conditions in
                                                  reconstructed_conditions]

            # Step 4: 验证原条件是否可以从重构条件推导
            verification_results = []  # 存储原条件与重构条件推导的结果

            # 用于存储当前批次的验证请求和对应的索引
            verification_prompts = []
            idx_mapping = []  # 用于存储每个条件的索引
            conditions_mapping = []  # 用于存储条件以便后续使用

            # 对于每个原条件和重构条件进行验证
            for idx, (conditions, recon_conditions) in enumerate(
                    zip(flattened_original_conditions, flattened_reconstructed_conditions)):

                for condition in conditions:
                    verification_prompt = (
                        f"Given a candidate condition:\"{condition}\"\n"
                        f"Here is a condition list:\"{', '.join(recon_conditions)}\"\n"
                        "From a mathematical point of view, can this candidate condition be deduced from the condition list? "
                        "Please illustrate your reason and answer \"yes\" or \"no\"."
                    )
                    # 将请求和对应的索引添加到批次中
                    verification_prompts.append(verification_prompt)
                    idx_mapping.append(idx)
                    conditions_mapping.append(condition)  # 存储与请求相关的条件

            # 批量处理验证请求，统一送入模型
            verification_outputs = llm.generate(verification_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_tokens,
            ))

            # 处理检验结果
            for output_text, idx, condition in zip(verification_outputs, idx_mapping, conditions_mapping):
                result_text = output_text.outputs[0].text if output_text else "No output"

                # 将条件与结果组成一个元组并存储于对应问题的验证结果中
                if idx < len(verification_results):
                    verification_results[idx][1].append((condition, result_text))
                else:
                    # 做防御性编程，确保不会超出范围
                    verification_results.append((idx, [(condition, result_text)]))

            # Step 5: 验证重构条件是否可以从原条件推导
            reconstructed_verification_results = []  # 存储重构条件与原条件推导的结果

            verification_prompts = []  # 存储当前批次的验证请求
            idx_mapping = []  # 存储索引
            recon_conditions_mapping = []  # 用于存储条件以便后续使用

            # 对于每个重构条件和原条件进行验证
            for idx, (recon_conditions, conditions) in enumerate(
                    zip(flattened_reconstructed_conditions, flattened_original_conditions)):

                for recon_condition in recon_conditions:
                    verification_prompt = (
                        f"Given a candidate condition:\"{recon_condition}\"\n"
                        f"Here is a condition list:\"{', '.join(conditions)}\"\n"
                        "From a mathematical point of view, can this candidate condition be deduced from the condition list? "
                        "Please illustrate your reason and answer \"yes\" or \"no\"."
                    )
                    # 将请求和对应的索引添加到批次中
                    verification_prompts.append(verification_prompt)
                    idx_mapping.append(idx)
                    recon_conditions_mapping.append(recon_condition)  # 存储与请求相关的条件

                    # 批量处理验证请求，统一送入模型
            verification_outputs = llm.generate(verification_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_tokens,
            ))

            # 处理检验结果
            for output_text, idx, recon_condition in zip(verification_outputs, idx_mapping, recon_conditions_mapping):
                result_text = output_text.outputs[0].text if output_text else "No output"
                # 将重构条件与结果组成一个元组并存储于对应问题的重构条件验证结果中
                if idx < len(reconstructed_verification_results):
                    reconstructed_verification_results[idx][1].append(
                        (recon_conditions, result_text))  # 此处需要确保用的是对应的条件
                else:
                    # 做防御性编程，确保不会超出范围
                    reconstructed_verification_results.append((idx, [(recon_conditions, result_text)]))

            # Step 6: 比较原问题和重构问题
            comparison_results = []  # 存储比较结果

            comparison_prompts = []  # 存储比较请求
            comparison_idx_mapping = []  # 存储索引

            # 对于每个原问题和重构问题进行比较
            for idx, (original, reconstructed) in enumerate(zip(original_prompts, reconstructed_questions)):
                comparison_prompt = (
                    f"Q1:\"{original}\"\n"
                    f"Q2:\"{reconstructed}\"\n\n"
                    "From a mathematical point of view, are these two problems asking the same thing at the end? "
                    "Please illustrate your reason and answer \"yes\" or \"no\"."
                )
                comparison_prompts.append(comparison_prompt)
                comparison_idx_mapping.append(idx)

                # 批量处理比较请求，统一送入模型
            comparison_outputs = llm.generate(comparison_prompts, SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens_per_call,
                n=1,
                stop=stop_tokens,
            ))

            # 处理比较结果
            for output_text, idx in zip(comparison_outputs, comparison_idx_mapping):
                comparison_result_text = output_text.outputs[0].text if output_text else "No output"
                # 将结果作为元组存储（索引、原问题、重构问题、比较结果）
                comparison_results.append(
                    (idx, original_prompts[idx], reconstructed_questions[idx], comparison_result_text))

            processed_conditions = []  # 存储最终输出条件信息的列表

            # Step 7: 处理原问题条件的情况
            new_conditions_prompts = []  # 用于批量生成新条件的提示
            prompt_indices = []  # 存储提示的索引

            for i, (original_results, recon_results) in enumerate(
                    zip(verification_results, reconstructed_verification_results)):
                original_problems_conditions = []
                reconstruction_problems_conditions = []

                # 检查原问题条件的判断结果
                for condition, result_text in original_results[1]:
                    if "no" in result_text.lower():  # 查找有问题的条件
                        original_problems_conditions.append((condition, result_text))

                        # 检查重构问题条件的判断结果
                for recon_condition, result_text in recon_results[1]:
                    if "no" in result_text.lower():  # 查找有问题的重构条件
                        reconstruction_problems_conditions.append((recon_condition, result_text))

                        # 分析判别结果
                comparison_result = comparison_results[i][3]  # 获取当前问题的比较结果

                # 如果有问题，则生成新条件的提示
                if original_problems_conditions or reconstruction_problems_conditions or "no" in comparison_result.lower():
                    new_conditions_prompt = (
                        f"Based on the original problem: \"{original_prompts[i]}\" and "
                        f"the reconstructed problem: \"{reconstructed_questions[i]}\".\n"
                        "The original conditions and their verification results are as follows:\n"
                    )

                    if original_problems_conditions:
                        for condition, result in original_problems_conditions:
                            new_conditions_prompt += f"- Original Condition: \"{condition}\", Verification: \"{result}\"\n"
                    else:
                        new_conditions_prompt += "- No problematic original conditions found.\n"

                    new_conditions_prompt += "The reconstructed conditions and their verification results are as follows:\n"

                    if reconstruction_problems_conditions:
                        for recon_condition, result in reconstruction_problems_conditions:
                            new_conditions_prompt += f"- Reconstructed Condition: \"{recon_condition}\", Verification: \"{result}\"\n"
                    else:
                        new_conditions_prompt += "- No problematic reconstructed conditions found.\n"

                    new_conditions_prompt += (
                        f"The comparison result between the original and reconstructed problem is: \"{comparison_result}\".\n"
                        "Please generate a new set of conditions that would ensure correctness and compatibility, "
                        "considering the issues found in the original conditions, reconstructed conditions, and their comparison."
                    )

                    # 添加到批量生成的新条件提示列表中
                    new_conditions_prompts.append(new_conditions_prompt)
                    prompt_indices.append(i)  # 记录原始索引

            # 批量生成新的条件
            if new_conditions_prompts:
                new_conditions_outputs = llm.generate(new_conditions_prompts, SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,  # 批量生成数量
                    stop=stop_tokens,
                ))

                # 保存新的条件
                for output, idx in zip(new_conditions_outputs, prompt_indices):
                    processed_conditions.append({
                        "original_question": original_prompts[idx],
                        "reconstructed_question": reconstructed_questions[idx],
                        "conditions": output.outputs[0].text if output else "No output"
                    })

                    # 处理没有问题的情况
            for i, (original_results, recon_results) in enumerate(
                    zip(verification_results, reconstructed_verification_results)):
                original_problems_conditions = []
                reconstruction_problems_conditions = []

                # 检查原问题条件的判断结果  
                for condition, result_text in original_results[1]:
                    if "no" in result_text.lower():  # 查找有问题的条件  
                        original_problems_conditions.append((condition, result_text))

                        # 检查重构问题条件的判断结果  
                for recon_condition, result_text in recon_results[1]:
                    if "no" in result_text.lower():  # 查找有问题的重构条件  
                        reconstruction_problems_conditions.append((recon_condition, result_text))

                # 再次检查，以确保没有问题才保存原问题条件
                if not (original_problems_conditions or reconstruction_problems_conditions or "no" in
                        comparison_results[i][3].lower()):
                    processed_conditions.append({
                        "original_question": original_prompts[i],
                        "reconstructed_question": reconstructed_questions[i],
                        "conditions": ", ".join(flattened_original_conditions[i])  # 保存原条件
                    })

            conditions_prompts = []
            for idx, processed in enumerate(processed_conditions):
                prompt = (
                    f"Based on the original question: \"{processed['original_question']}\" and "
                    f"the reconstructed question: \"{processed['reconstructed_question']}\"\n"
                    f"The conditions considered are: \"{processed['conditions']}\"\n"
                    "Please refine or summarize these conditions into a clear and concise format."
                )
                conditions_prompts.append((idx, prompt))  # 将构建的序号和提示以元组的形式添加到列表

            current_prompts = conditions_prompts


            # 提取当前提示，并使用指定的采样参数从 LLM 生成输出。
        # SamplingParams 指定生成时的温度、顶点分布概率等。
        prompts = [item[1] for item in current_prompts]
        outputs = llm.generate(prompts, SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
            stop=stop_tokens,
        ))

        # 对输出按请求 ID 进行排序，并提取文本。
        # 确保输出数量与当前提示数量一致，以进行后续处理。
        outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
        assert len(outputs) == len(current_prompts)

        # process all outputs
        # 根据输出和提示类型更新 remain_prompts 和 remain_codes。
        # 处理不同的提示类型，以决定代码是否需要执行。
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            # 用于去除输出字符串末尾的空白字符（如空格和换行符）
            output = output.rstrip()
            # 将生成的输出附加到当前的查询字符串中，形成完整的查询内容。
            query += output
            # 如果 prompt_type 是 "pal"，则将 (i, query) 作为元组添加到 remain_prompts 列表中，表示这个查询仍然需要进一步处理
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            # 如果 prompt_type 是 "cot"，则将 (i, query) 添加到 end_prompts 列表中，表示这个查询已经完成，不需要进一步处理
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            # 检查输出是否不包含 "boxed" 并且以 "```" 结尾。这通常意味着输出是一个代码块，但不是以 "boxed" 格式返回的。
            elif ("boxed" not in output and output.endswith("```")):
                program = extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            # 如果以上条件都不满足，说明输出的格式不符合预期，则将 (i, query) 添加到 end_prompts 列表中，表示这个查询已经完成
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        # 调用 executor.batch_apply(remain_codes) 来执行剩余的代码，并获取结果。
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    ans_split = "<|assistant|>" if args.use_train_prompt_format else "Question:"
    codes = [prompt.split(ans_split)[-1].strip() for _, prompt in end_prompts]

    # extract preds
    results = [run_execute(executor, code, args.prompt_type) for code in codes]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i*args.n_sampling: (i+1)*args.n_sampling]
        result = results[i*args.n_sampling: (i+1)*args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]

        sample.pop('prompt')
        sample.update({'code': code, 'pred': preds, 'report': reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    save_jsonl(all_samples, out_file)

    result_str = evaluate(samples=all_samples, data_name=args.data_name, prompt_type=args.prompt_type, execute=True)
    result_str += f"\nTime use: {time_use:.2f}s"
    time_str = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    result_str += f"\nTime use: {time_str}"

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}.metrics"), "w") as f:
        f.write(result_str)

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
