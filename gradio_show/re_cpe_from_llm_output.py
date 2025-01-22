# ## 将使用llm得到的output提取出cpe信息，然后将cpe转换为json格式。

import os
import re


def extract_ai_output(text):
    pattern_section = r"### AI 输出:(.*?)(?=###|$)"
    sections = re.findall(pattern_section, text, re.DOTALL)
    return sections

filepath = '/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/output_qwen72_cpe_new2_prompt_eng/output_66.txt'
with open(filepath, 'r', encoding='utf-8') as file:
    text = file.read()
# ai_sections = extract_ai_output(text)
# print(ai_sections)

def extract_first_cpe_string(sections):
    pattern_cpe = r"cpe:\/([ahoil]):([\w-]+):([\w-]+|\*):([<>=]*[\w\.-]*|\*):([\w\.-]*|\*):([\w\.-]*|\*):([\w\.-]*|\*)"
    for section in sections:
        match = re.search(pattern_cpe, section)
        if match:
            return 'cpe:/{}:{}:{}:{}:{}:{}:{}'.format(*match.groups())
    return None
# first_cpe = extract_first_cpe_string(ai_sections)
# print(first_cpe)


def process_files(directory, output_file):
    file_paths =[]
    first_cpes = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            # file_paths.append(filepath)
            # print(len(file_paths)) #100
            #print(filepath)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            ai_sections = extract_ai_output(text)
            first_cpe = extract_first_cpe_string(ai_sections)
            if first_cpe:
                first_cpes.append(first_cpe)
            print(filepath,first_cpe)
            # print(len(first_cpes))


    # Write all first CPEs to a single output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for cpe in first_cpes:
            outfile.write(cpe + '\n')


# 设置目录路径和输出文件路径
directory_path = "/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/output_qwen72_cpe_new2_prompt_eng"
output_file_path = "/root/data/wjy/vip_vul_pro/RAG_VIP_VULN/llm/first_cpes.txt"
process_files(directory_path, output_file_path)



