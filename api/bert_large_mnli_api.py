from transformers import pipeline
import torch



def get_response(prompt, labels):
    # automatically choose a device:
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\n ** device selected = {device}! \n")


    # load model
    classifier = pipeline(
        task="zero-shot-classification",
        model="facebook/bart-large-mnli",
        device_map="auto", 
        batch_size=2,
        )

    sequence_to_classify = prompt
    candidate_labels = labels

    result = classifier(sequence_to_classify, candidate_labels)
    return result



















#{'labels': ['travel', 'dancing', 'cooking'],
# 'scores': [0.9938651323318481, 0.0032737774308770895, 0.002861034357920289],
# 'sequence': 'one day I will see the world'}


# '''
# # If more than one candidate label can be correct, pass multi_label=True to calculate each class independently:
# candidate_labels = ['travel', 'cooking', 'dancing', 'exploration']
# classifier(sequence_to_classify, candidate_labels, multi_label=True)
# #{'labels': ['travel', 'exploration', 'dancing', 'cooking'],
# # 'scores': [0.9945111274719238,
# #  0.9383890628814697,
# #  0.0057061901316046715,
# #  0.0018193122232332826],
# # 'sequence': 'one day I will see the world'}
# '''







# # pose sequence as a NLI premise and label as a hypothesis
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch

# # load model
# nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
# tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')



# # automatically choose a device:
# device = None
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     if torch.backends.mps.is_built():
#         device = torch.device("mps")
# else:
#     device = torch.device("cpu")
# print(f"\n *** device selected = {device}!\n")

# news = '''Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.  Loretta Lynch aka Eric Holder in a skir.'''
# premise = news

# # label from here:
# candidate_labels = ['political', 'sport', 'entertainment']
# hypothesis = f'This example is {candidate_labels}.'


# # run through model pre-trained on MNLI
# x = tokenizer.encode(premise, hypothesis, return_tensors='pt',
#                      truncation_strategy='only_first')
# logits = nli_model(x.to(device))[0]

# # we throw away "neutral" (dim 1) and take the probability of
# # "entailment" (2) as the probability of the label being true 
# entail_contradiction_logits = logits[:,[0,2]]
# probs = entail_contradiction_logits.softmax(dim=1)
# prob_label_is_true = probs[:,1]
