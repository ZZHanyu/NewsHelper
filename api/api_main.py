import bert_large_mnli_api
import openai_api
import tinyllama_api



def build_prompt(raw_news):



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
    

def call_apis(prompt):
    