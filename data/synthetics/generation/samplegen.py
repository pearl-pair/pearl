import torch
import json
import time
import os
import re
import nltk
import random
import argparse
from collections import defaultdict

# Download English stopwords for filtering later
nltk.download('stopwords')
from nltk.corpus import stopwords
from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import Namespace


# === ARGUMENT DEFINITIONS ===
args = Namespace(**{
    'iter': 15,                      # Number of outer iterations for generation per persona
    'cal': 15,                       # Number of calendar-type samples to generate per iteration
    'msg': 20,                       # Number of message-type samples to generate per iteration
    'con': 10,                       # Number of contact-type samples to generate per iteration
    'output': "output{}.json"        # Output filename format (will be formatted with persona ID)
})

# Specify which GPU to use (index in CUDA device list)
args.gpu = 0

# List of persona indices to generate data for
args.persona_list = [0,1]

# Maximum number of raw samples to generate per persona
args.persona_max_samples = [200,300]

# Extract the number of iterations for convenience
max_iterations = args.iter


# === MODEL LOADING ===
model_name = "Qwen/Qwen2.5-14B-Instruct"
print("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map=f"cuda:{args.gpu}"
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
stop_words = set(stopwords.words("english"))


# === UTILITY FUNCTIONS ===

# Determine passage type from required fields
def is_valid_passage_type(sample):
    passage = sample.get("passage", {})
    if all(key in passage for key in ["title", "desc", "date", "location", "attendees"]):
        return "calendar"
    elif all(key in passage for key in ["body", "from", "relation", "group"]):
        return "message"
    elif all(key in passage for key in ["name", "relation", "group", "email"]):
        return "contact"
    print("INVALID PASSAGE:", json.dumps(passage, indent=2))
    return None

# Calculate n-gram overlap between query and passage (for filtering)
def ngram_overlap(text1, text2, n=2, use_stopwords=True):
    def tokenize_filtered(s):
        tokens = re.findall(r"\b\w+\b", s.lower())
        if use_stopwords:
            return [t for t in tokens if t not in stop_words]
        return tokens

    def get_ngrams(tokens, n):
        return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    tokens1 = tokenize_filtered(text1)
    tokens2 = tokenize_filtered(text2)
    ngrams1 = get_ngrams(tokens1, n)
    ngrams2 = get_ngrams(tokens2, n)

    return len(ngrams1 & ngrams2)

# Convert structured passage dict to string for lexical checks
def flatten_passage(passage, passage_type):
    if passage_type == "calendar":
        return passage["title"] + passage["desc"]
    elif passage_type == "message":
        return passage["body"]
    elif passage_type == "contact":
        return passage["name"] + passage["relation"] + passage["group"]
    return ""

# Simple rule-based check for past tense in calendar description
def contains_past_tense(passage_obj):
    desc = passage_obj.get("desc", "").lower()
    past_tense_verbs = {
        "met", "joined", "spent", "had", "discussed", "celebrated", "attended",
        "visited", "talked", "played", "worked", "called", "messaged", "emailed",
        "texted", "walked", "drove", "went", "hosted", "presented", "helped",
        "studied", "danced", "ate", "watched", "saw", "listened", "replied",
        "asked", "gave", "shared", "enjoyed", "purchased", "completed"
    }
    return any(re.search(rf"\\b{verb}\\b", desc) for verb in past_tense_verbs)


# Extract text as json
def extract_samples_from_text(text):
    samples = []
    entries = re.split(r"(?:^|\n)\s*QUERY:", text)
    for entry in entries[1:]:
        query = {}
        passage = {}
        lines = entry.strip().splitlines()
        mode = None
        first_line = True
        for line in lines:
            line = line.strip()
            if first_line:
                query["txt"] = line.strip()
                mode = "query"
                first_line = False
            elif line.startswith("- CT:") and mode == "query":
                query["ct"] = line.replace("- CT:", "").strip()
            elif line.startswith("- TITLE:"):
                mode = "calendar"
                passage["title"] = line.replace("- TITLE:", "").strip()
            elif line.startswith("- DESC:") and mode == "calendar":
                passage["desc"] = line.replace("- DESC:", "").strip()
            elif line.startswith("- DATE:") and mode == "calendar":
                passage["date"] = line.replace("- DATE:", "").strip()
            elif line.startswith("- LOCATION:") and mode == "calendar":
                passage["location"] = line.replace("- LOCATION:", "").strip()
            elif line.startswith("- ATTENDEES:") and mode == "calendar":
                attendees = line.replace("- ATTENDEES:", "").strip()
                passage["attendees"] = [att.strip() for att in attendees.split(",") if att.strip()]
            elif line.startswith("- FROM:"):
                mode = "message"
                passage["from"] = line.replace("- FROM:", "").strip()
            elif line.startswith("- BODY:") and mode == "message":
                passage["body"] = line.replace("- BODY:", "").strip()
            elif line.startswith("- RELATION:") and mode in ["message", "contact"]:
                passage["relation"] = line.replace("- RELATION:", "").strip()
            elif line.startswith("- GROUP:") and mode in ["message", "contact"]:
                passage["group"] = line.replace("- GROUP:", "").strip()
            elif line.startswith("- EMAIL:") and mode == "contact":
                passage["email"] = line.replace("- EMAIL:", "").strip()
            elif line.startswith("- NAME:"):
                mode = "contact"
                passage["name"] = line.replace("- NAME:", "").strip()
            elif line.startswith("- CT:") and mode in ["calendar", "message", "contact"]:
                passage["ct"] = line.replace("- CT:", "").strip()

        if query and passage:
            samples.append({"query": query, "passage": passage})
    return samples


# Generate samples
def generate_samples(messages, total_elapsed):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("Generating response...")
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        do_sample=True,
        temperature=1.5,
        top_k=100,
        max_new_tokens=4096
    )
    generation_time = time.time() - start_time

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Raw model output:\n", response_text)
    gen_minutes, gen_seconds = divmod(int(generation_time), 60)
    cum_minutes, cum_seconds = divmod(int(total_elapsed + generation_time), 60)
    log_msg = f"Time taken for generation: {gen_minutes}m {gen_seconds}s (Total: {cum_minutes}m {cum_seconds}s)"
    print(log_msg)
    with open("run.log", "a") as f:
        f.write(log_msg + "\n")
    
    return response_text, generation_time 

for idx_p, persona in enumerate(args.persona_list):
    # Load Persona 
    persona_path = "personas.txt"
    with open(persona_path, "r", encoding="utf-8") as pf:
        raw = pf.read().strip().split("- **Index**: ")
        personas = {}
        for chunk in raw[1:]:
            lines = chunk.strip().splitlines()
            idx = int(lines[0])
            text = "- **Index**: " + "\n".join(lines)
            personas[idx] = text
    
    persona_text = personas.get(persona)
    if not persona_text:
        raise ValueError(f"Persona with index {persona} not found in {persona_path}")

    existing_passages = set()
    existing_queries = set()
    main_messages = [
        {"role": "system", "content": "You are an assistant generating query-passage examples in plain text format."},
    ]
    output_file = args.output.format(persona)
    
    # If an output file already exists, load and deduplicate previously generated samples
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                for item in json.load(f):
                    existing_queries.add(item["query"]["txt"])
                    passage_type = is_valid_passage_type(item)
                    existing_passages.add(flatten_passage(item["passage"], passage_type))
            except json.JSONDecodeError:
                print("Failed to load existing data.")
    
    total_start = time.time()
    cumulative_time = 0
    main_prompts = {}
    cum_all_samples = []
    raw_text_samples_by_type = {"calendar": [], "message": [], "contact": []} 
    prev_iter_texts_by_type = {"calendar": [], "message": [], "contact": []}


    for outer_iter in range(max_iterations):
        print(f"\n=== Iteration {outer_iter+1} ===")
        all_new_samples = []

        for target_type in ["calendar", "message", "contact"]:
            type_to_num_samples = {"calendar": args.cal, "message": args.msg, "contact": args.con}
            print(f"[{persona}] Generating {type_to_num_samples[target_type]} {target_type} samples for Iteration {outer_iter+1}...")
    
            if outer_iter == 0:
                with open(f"{target_type}_prompt.txt", "r", encoding="utf-8") as p:
                    prompt = p.read().replace("{N}", str(type_to_num_samples[target_type])).replace("{persona}", persona_text)
                    prompt += f"Output exactly {str(type_to_num_samples[target_type])} samples. Each sample must start with 'QUERY:' and include passage fields properly. No JSON."
                    prompt += " If any lemma from the query appears in the passage, regenerate that pair."
                    main_prompts[target_type] = prompt
                    messages = [*main_messages, {"role": "user", "content": prompt}]
            else:
                prompt = main_prompts[target_type] + "Never generate word-to-word duplicates or semantically similar examples with the previously generated examples or responses. In other words, be diverse on vocabulary selections or word choices."
                messages = [*main_messages]

                for prev_type in ["calendar", "message", "contact"]:
                    for block in prev_iter_texts_by_type.get(prev_type, []):
                        messages.append({"role": "assistant", "content": block})
                messages.append({"role": "user", "content": prompt})

            new_samples_text, gen_time = generate_samples(messages, cumulative_time)
            cumulative_time += gen_time

            if new_samples_text:
                prev_iter_texts_by_type[target_type] = [new_samples_text]
                
                with open(f"raw_samples_{persona}.txt", "a", encoding="utf-8") as raw_f:
                    raw_f.write(new_samples_text + "\n")   

        current_total_samples = sum(
            block.count("QUERY:") for blocks in raw_text_samples_by_type.values() for block in blocks
        )              
        print(f"Current raw sample count: {current_total_samples}") 
        if current_total_samples >= args.persona_max_samples[idx_p]:
            print(f"[STOP] Persona {persona} reached max raw sample count: {current_total_samples}")
            break 
          

    print("\n=== Post-processing raw samples ===")
    with open(f"raw_samples_{persona}.txt", "r", encoding="utf-8") as raw_f:
        raw_text = raw_f.read()

    parsed_samples = extract_samples_from_text(raw_text)
    filtered_samples = []

    for sample in parsed_samples:
        query_txt = sample["query"]["txt"]
        passage_type = is_valid_passage_type(sample)

        if not passage_type:
            print(f"[FILTERED] Invalid structure: {query_txt}")
            continue

        if passage_type == "message" and "last" in sample["passage"].get("body", "").lower():
            print(f"[FILTERED] 'last' found in message body: {query_txt}")
            continue

        if passage_type == "calendar" and contains_past_tense(sample["passage"]):
            print(f"[FILTERED] Contains past tense: {query_txt}")
            continue

        passage_txt = flatten_passage(sample["passage"], passage_type)
        if ngram_overlap(query_txt, passage_txt) >= 3:
            print(f"[FILTERED] High lexical overlap: {query_txt}")
            continue

        filtered_samples.append(sample)

    # Final JSON output save
    with open(output_file, "a", encoding="utf-8") as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    torch.cuda.empty_cache()
        
    # === Final Time Log ===
    total_time = time.time() - total_start
    hours, remainder = divmod(int(total_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal generation time: {hours}h {minutes}m {seconds}s")