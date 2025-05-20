import torch
import argparse
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Command-line argument for number of personas ===
parser = argparse.ArgumentParser(description="Generate new personas using Qwen2.5")
parser.add_argument("--num", type=int, default=5, help="Number of new personas to generate")
args = parser.parse_args()

# === Load model and tokenizer ===
model_name = "Qwen/Qwen2.5-14B-Instruct-1M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# === Load existing personas ===
with open("personas.txt", "r", encoding="utf-8") as f:
    existing_personas = f.read()

# === Create prompt ===
prompt = f"""
Below is a list of structured persona entries. Each entry contains rich biographical details, hobbies, relationships, and locations. 
Generate {args.num} **new** personas that match the tone, structure, and formatting style of the existing entries, but have **distinct personalities, occupations, hobbies, and relationships**.

Each generated persona **must satisfy the following rules**:

1. Must include exactly one **mother** and one **father** relationship (with appropriate names and brief details).

2. Include at least **5 hobbies**, **5 frequent locations**, and **8 relationships**.

3. At least **5 relationships must include nicknames** using either:
   - Nicknames not derived from names (e.g., *jellybean*, *darling*, *homie*, *chief*, *my love*, *champ*, etc.), or
   - Abbreviated forms of names (e.g., *Stephen → Steve*, *Katherine → Kathy*)

4. Relationships should be diverse (family, friends, colleagues, therapist, etc.) and described with naturalistic, specific traits or interactions.

5. DO NOT reuse hobbies that have already been generated in the previous personas. Avoid common hobbies such as: hiking, attending music festivals, yoga, birdwatching, practicing mindfulness, attending art exhibitions. 

6. DO NOT reuse names that have already been included in previous personas. Avoid common names such as: Olivia, Liam, Theo, Alex, Mary, Robert, David. 

7. There should be at least 1 teenager, 1 young adult, 1 middle-aged adult, and 1 elder persona. 

Below are personas that have already been generated:

{existing_personas}

Now continue the list by adding personas with - **Index**:
"""

# === Generate new personas ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=8000,
    temperature=0.91,
    top_k=110,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# === Extract new entries only (starting at Index 38 or more) ===
new_personas = "\n".join(re.findall(r"- \*\*Index\*\*: \d+.*", generated_text, re.DOTALL))

# Save to file
with open("generated_personas.txt", "w", encoding="utf-8") as f:
    f.write(new_personas.strip())

print(f"Successfully generated {args.num} new personas.")
