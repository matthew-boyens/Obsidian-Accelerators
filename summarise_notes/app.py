# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.10.1 64-bit
#     language: python
#     name: python3
# ---

# %%
import os
import openai
import re
import numpy as np
import markdown

from tqdm import tqdm
from bs4 import BeautifulSoup


MODEL_COST = {"text-davinci-002":0.0200}

DAVANCI_COST = 0.0200

# %%
with open("secret.txt") as f:
    api = f.read()

# %%
openai.api_key = api

# %%
MAX_TOKENS = 4000
TOKENS_PER_CHAR = 4
OUTPUT_TOKENS = 500
MAX_INPUT_CHARS = (MAX_TOKENS - OUTPUT_TOKENS)*TOKENS_PER_CHAR

MAX_INPUT_CHARS


# %%
def gpt_summary_on_podcast(prompts,filepath):
    """Takes Prompts in format [["Prompt","Model"]] and applies them to the file found at filename """

    with open(filepath) as f:
        file = f.read()

        html = markdown.markdown(file)

        soup = BeautifulSoup(html)
        document = []
        ## specific to podcast format
        for para in soup.find_all("p")[1:]:
            document.append(para.get_text())
    print("Loaded Document")
    print(f"Estimated cost: {(sum([len(para) for para in document])/4000)* len(prompts)*DAVANCI_COST}")

    ## Running results
    results = []
    print("Api called")
    for para in tqdm(document):
        for prompt, model in prompts:
            results.append(openai.Completion.create(
            model=model,
            prompt=prompt+para,
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            ))
    print("GPT3 Run")

    print(f"Actual cost: {(sum([result['usage']['total_tokens'] for result in results])/1000) * 0.0200}")

    ## Clean up results and make list into 2D array grouped by paragraph
    results_flat = np.reshape(results,-1)
    results_text = [result['choices'][0]['text'] for result in results_flat]

    results_text_clean = [re.sub("^\n\n","",result) for result in results_text]
    results_grouped = np.reshape(results_text_clean,(-1,len(prompts)))

    print("Results Cleaned")


    
    ## Save to .md file
    alias_break_count = 0
    result_i =0
    with open(filepath) as f_old, open(filepath[:-3] + " + GPT3.md", "w") as f_new:
        for line in f_old:

            if 'Transcript:' in line:
                f_new.write(f"GPT-3 Summary:\n")
                ## Tech debt - prompt titles are not coupled to prompt results
                f_new.write(f"#### Themes:\n{results_grouped[result_i][0]}\n")
                f_new.write(f"#### Key points:\n{results_grouped[result_i][1]}\n")
                f_new.write(f"#### Notes:\n\n")
                f_new.write("##### ")
                result_i +=1
            elif "---" in line:
                if alias_break_count>1:
                    f_new.write("####\n")
                else:
                    alias_break_count += 1  
            elif "Status::" in line:
                line = "- Status:: #📥/🟧 "
            f_new.write(line)

    print("Saved to file")

    return results_grouped
    

        
