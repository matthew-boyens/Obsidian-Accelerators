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
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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


MODEL_INFO = {"text-davinci-002":{"cost":0.0200, "max_tokens":4000}}

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
class Prompt:

    def __init__(self,short,full,model,on):

        self.short = short
        self.full = full
        self.model = model
        self.on = on

    def __eq__(self,other):
        if isinstance(other,Prompt):
            return self.short == other.short
        else:
            return False

    def __repr__(self) -> str:
        return f" Prompt({self.short},{self.model},{self.on})"
    def __str__(self) -> str:
        return f" Prompt({self.short},{self.model},{self.on})"


# %%
class Document:

    ## Each paragraph must start with --- and contain "Transcript" -> directly above this the summary text will be injected
    
    def __init__(self,filepath,format="podcast"):

        self.filepath = filepath
        self.format = format

        self.paragraphs = []
        self.doc_results = dict()
        self.doc_cost = 0
        self.saved = False
        self.processed_prompts = []

        with open(filepath) as f:
            file = f.read()

            html = markdown.markdown(file)
            self.soup = BeautifulSoup(html)
            
        
            
            ## specific to podcast format
            if self.format == "podcast":
                ## Append first paragraph element (assumes first element is aliase so that is skipped add)
                    text = self.soup.find("h2").find_next("p").find_next("p").get_text()
                    if text:
                        self.paragraphs.append(dict(text = text,total_cost=0))
                    ## don't include last element as no text is after it
                    for text in [hr.find_next("p").get_text() for hr in self.soup.find_all("hr")[:-1]]:
                        if text:
                            self.paragraphs.append(dict(text = text,total_cost=0))
#                 except:
#                     print("Missing element")
#                     print(self.soup)
            else:
                raise Exception(f"{self.format} is not supported")
    
    def estimated_cost(self,prompts):
        """Given prompts run on text takes up most of the cost, given an estimate off the cost"""

        tmp = []
        for para in self.paragraphs:
            for prompt in prompts:
                if prompt.on == "text":
                    tmp.append(len(para["text"])*MODEL_INFO[prompt.model]["cost"])
        
        return f"Estimated cost: {sum(tmp)/4000}"


    def total_cost(self):

        cost = (sum([para["total_cost"] for para in self.paragraphs]) + self.doc_cost)
        return f"Actual cost: {cost}"

    
    @staticmethod
    def run_open_ai(text,prompt):
         ## Running results

        ## Limit prompt length
        combined_prompt = (prompt.full+text)[:min(3750*4,len(prompt.full+text))]
        result = openai.Completion.create(
            model=prompt.model,
            prompt=combined_prompt,
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )

        ## Removes 2 new lines usually there between query and result
        text = re.sub("^\n\n","",result['choices'][0]['text'])
        cost = (result["usage"]["total_tokens"]/1000)*MODEL_INFO[prompt.model]['cost']

        return text,cost





    def process(self,prompts = [ Prompt("Main themes",
                                    "Summarise into top themes in format \n'- [[<theme>]]\n' for the following text:\n\n","text-davinci-002","Key Points"),
                                    Prompt("Summary", "Abstractively Summarise main concepts within a short concise paragraph the following text:\n\n","text-davinci-002","Key Points"),
                                    #Prompt("Summary","Summarise main concepts within a short concise paragraph the following text:\n\n","text-davinci-002","Key Points"),
                                    Prompt("Key Points","Summarise into key points in format \n'- point\n' for the following text:\n\n","text-davinci-002","text")]):

       
        print(self.estimated_cost(prompts))




        for prompt in prompts:

            if prompt.on == "text":
                for para in tqdm(self.paragraphs):
                    text, cost = Document.run_open_ai(para['text'],prompt)
                    para[prompt.short] = text
                    para['total_cost'] += cost
            
        ## Document level/summary results - done as a separate loop to prevent summary prompt breaking if text not available


        for prompt in tqdm(prompts):
            if prompt.on != "text":

                segments = []
                include_segments = []

                for para in self.paragraphs:
                    segments.extend(para[prompt.on].split("\n"))

                for segment in segments:
                     # remove segmments that are over 200 in length to ensure tokens don't blow out
                    if len(segment) < 200:
                        include_segments.append(segment)
                #sample 75 key points as this will only use 75*(200/4) = 3750, currently sample is just random but might need to be stratified across paragraph to ensure coverage
                ## Tech Debt - 75 is based on 4000 tokens, need to do this per model applied for this prompt. Instead make this depend on Prompt which tells us what model is run.
                sampled_points = np.random.choice(include_segments,min(75,len(include_segments)))

                text, cost = Document.run_open_ai(" ".join(sampled_points),prompt)
                self.doc_results[prompt.short] = text
                self.doc_cost += cost

            #Creating sort order based on when prompt was called
            if prompt not in self.processed_prompts:
                self.processed_prompts.append(prompt)


        print(self.total_cost())

    def save(self):

            alias_break_count = 0
            self.para_i = 0
            with open(self.filepath) as f_old, open(self.filepath[:-3] + " + GPT3.md", "w") as f_new:
                for line in f_old:
                    
                    if "Status::" in line:
                        line = "- Status:: #📥/🟧\n"

                    elif "Highlights" in line:

                        f_new.write(f"### GPT-3 Summary:\n")

                        ### Document level results

                        for heading in self.doc_results:
                            f_new.write(f"#### {heading}:\n{self.doc_results[heading]}\n")

                        
                    elif 'Transcript:' in line:

                        for prompt in self.processed_prompts:

                            if prompt.on == "text":
                                f_new.write(f"#### {prompt.short}:\n{self.paragraphs[self.para_i][prompt.short]}\n")

                        f_new.write(f"#### Notes:\n\n")
                        f_new.write("##### ")
                        self.para_i +=1
                    elif "---" in line:
                        if alias_break_count>1:
                            f_new.write("####\n")
                        else:
                            alias_break_count += 1  
                    
                    f_new.write(line)

                if self.para_i == len(self.paragraphs) -1:
                    self.incorrect_aligment = False
                else:
                    self.incorrect_aligment = True

                self.saved = True

    def overwrite(self):

        if self.saved:
            os.remove(self.filepath)

            os.rename(self.filepath[:-3] + " + GPT3.md",self.filepath)
        else:
            print("Nothing to save")


# %%

def open_soup(filepath):
    
    with open(filepath) as f:
        file = f.read()

        html = markdown.markdown(file)
        return BeautifulSoup(html)


# %% [markdown]
# # Ideas
# - Create PodcastDocument(Document) class structure
#     - Create an abstract class document that has abstract classes
#         - load,
#         - enrich
#         - save
#     - These would then be defined within each inherited class
#     - This would allow the loading of different documents and running compute on them

# %%
filepath = "/Users/mboyens/Documents/SecondBrain/Readwise/Podcasts/294. Eugenics —  Flawed Thinking Behind Pushed Science.md"
# filepath = "/Users/mboyens/Documents/SecondBrain/Readwise/Podcasts/John Frusciante Returns, Part 1.md"


# %%
doc.soup.find("h2").find_next("p").find_next("p")

# %%
doc = Document(filepath)
doc.process()
doc.save()


# %%
len(doc.paragraphs)

# %%
doc.save()

# %%
doc.overwrite()

# %%
docs = []
errors = []
for file in tqdm(files):

    doc = Document(file)

    try:
        doc.process()
        doc.save()
        docs.append(doc)
    except:
        errors.append([doc])


# %%
for doc in docs:
    doc.overwrite()

# %%
overriden = []

for doc in docs:

    if filename not in overriden:
        print(filename)
        print("\n")
        print(doc.paragraphs[0],"\n",doc.paragraphs[-1])

        if bool(input("Overwrite?") or False):
            doc.overwrite()
            overriden.append(filename)

# %%
doc.process([Prompt("Summary V2", "Abstractively Summarise main concepts within a short concise paragraph the following text:\n\n","text-davinci-002","Key Points")])

# %%
doc.process([Prompt("Summary V2", "Abstractively Summarise main concepts within a short concise paragraph the following text:\n\n","text-davinci-002","Key Points")])



# %%
doc.saved=True

# %%
doc.overwrite()

# %%
## Analysis on Segment Lengths for Key Points
import pandas as pd

# %%
points = []
for para in doc.paragraphs.split("\n"):
    points.extend(para["Key Points"].split("\n"))

df = pd.DataFrame(points)
df.sort_values(['len'])

# %%
df[df['len'] <200].len.plot(kind='bar')

# %%
df[df.point.str.len()>200].point.str.len().describe()

# %%
len(points)

# %%
os.remove(filepath)

os.rename(filepath[:-3] + " + GPT3.md",filepath)

