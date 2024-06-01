from transformers import AutoTokenizer
import transformers 
import torch


def tinyllama_api(raw_news):
    model = "PY007/TinyLlama-1.1B-intermediate-step-240k-503b"
    tokenizer = AutoTokenizer.from_pretrained(model)

    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        device_map="auto",
    )

    # concat prompt
    # this is a one-shot prompt example:
    example_news = '''Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.  Loretta Lynch aka Eric Holder in a skir.'''
    
    instruction = "Classify what the labels of the news described in <News Context> (e.g., is politics, sports, entertainment) ?"
    
    example = f''' 
    <system>: - Instruction: "{instruction}".
        - News Context:: "{example_news}".
    <bot>: Answer: label is "Politics, Laws" .
    '''
    input_prompt = f'''
    <system>: Below is an instruction that describes a task. Write a answer that appropriately completes the instruction, answer format need same with example: 
    Example:<{example}>.
    <system>: - Instruction: "{instruction}".
        - News Context: "{raw_news}".
    <bot>: Answer: label is 
        '''

    sequences = pipeline(
        text_inputs=input_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        repetition_penalty=1.5,
        eos_token_id=tokenizer.eos_token_id,
        #max_length=1024,
    )


    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
    
    return sequences
        
