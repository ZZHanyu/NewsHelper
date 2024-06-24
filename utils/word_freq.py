from tqdm import tqdm
import re
import pandas as pd
import matplotlib.pyplot as plt

def _remove_empty_line(single_chunk):
    for index, row in single_chunk.iterrows():
        if pd.isnull(row['title']) or pd.isnull(row['text']) or pd.isnull(row['label']):
            single_chunk.drop(index, inplace=True)
        elif type(row['title']) != str or type(row['text']) != str or type(row['label']) != int:
            single_chunk.drop(index, inplace=True)
        elif len(row['title']) < 5 or len(row['text']) < 5:
            single_chunk.drop(index, inplace=True)
    return single_chunk


def simple_str_handler(single_string):
    '''
        return a word of bag
    '''
    single_string = single_string.lower()
    
    # 把标点符号用用空格替换  
    single_string = re.sub(r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]','',single_string).strip()

    single_string = single_string.split()

    return single_string


def csv_reader(freq_dict):
    chunks = pd.read_csv("/Users/taotao/Documents/GitHub/FYP/data/fake-news-classification/WELFake_Dataset.csv", chunksize=50)
    for _ in tqdm(range(10), desc="total counter progress...", leave=True):
        single_chunk = next(chunks)
        single_chunk = _remove_empty_line(single_chunk)

        for row in single_chunk.iterrows():
            word_bag = simple_str_handler(row[1]['title'] + row[1]['text'])
            word_freq_counter(freq_dict, word_bag)

    print(sorted(freq_dict.items(), key = lambda kv:(kv[1], kv[0])))



def word_freq_counter(freq_dict, word_bag):

    for single_word in word_bag:
        if freq_dict.get(single_word, None) == None:
            freq_dict[single_word] = 1
        else:
            freq_dict[single_word] += 1
        


def label_balance_checker():
    chunks = pd.read_csv("/Users/taotao/Documents/GitHub/FYP/data/fake-news-classification/WELFake_Dataset.csv", chunksize=50)
    truth = 0
    fake = 0

    try:
        while True:
            single_chunk = next(chunks)
            single_chunk = _remove_empty_line(single_chunk)

            if not isinstance(single_chunk, pd.DataFrame):
                break
            else:
                for row in single_chunk.iterrows():
                    if row[1]['label'] == 1:
                        truth += 1
                    else:
                        fake += 1
        
        
    except StopIteration:
        # draw dataset balance diagram
        categories = ['truth', 'fake']
        counts = [truth, fake]
        plt.figure()
        plt.bar(categories, counts, color='skyblue')
        plt.title('Balance Diagram')
        plt.xlabel('Categories')
        plt.ylabel('Values')
        plt.show()



if __name__ == "__main__":
    
    freq_dict = {}
    csv_reader(freq_dict)
    # news_context = '''Mexico presidential race roiled as leftist front-runner embraces right wing party,"MEXICO CITY (Reuters) - The front-runner in Mexico s 2018 election has embraced a small, socially conservative party in his bid for the presidency, sparking criticism among progressives that could splinter his support in what is expected to be a tight race. Earlier this week, two-time presidential runner-up Andres Manuel Lopez Obrador entered into a coalition with the Social Encounter Party (PES), a tiny party with religious roots that pushes an anti-gay and anti-abortion agenda. The coalition is led by Lopez Obrador s left-of-center MORENA party and also includes the socialist-leaning Labor Party. Lopez Obrador, who is leading in most polls, has been Mexico s best-known leftist since he served as mayor of the capital a decade ago. In previous elections he favored tie-ups with left-wing parties. Some political analysts said a coalition with social conservatives could provide Lopez Obrador with the margin he needs to prevail in a crowded field of candidates, including first-time independents. But cracks in his progressive base are emerging.  It looks like our rights will have to keep waiting,  said Lol Kin Castaneda, a leading gay rights activist who helped push Mexico City s approval of marriage equality in 2010. Mexico s Supreme Court ruled that gay marriage is lawful. Still, fewer than half of the country s 32 states have laws on the books that permit civil marriage for gays and lesbians, and individual same-sex couples in most states must fight to be recognized, often at great cost.  It s an error for MORENA to join forces with PES... In this alliance, PES wins,  Andres Lajous, a left-leaning writer and television pundit, posted on Twitter. Leo Zuckermann, a socially liberal commentator, suggested that Lopez Obrador, popularly known by his initials AMLO, is revealing his true colors by aligning himself with the PES.  People of the left, it s time to recognize that AMLO is a conservative on these issues,  he wrote in newspaper Excelsior. Lopez Obrador, like many Latin American leftists, has traditionally focused on fighting poverty and graft, not divisive social issues like gay rights or a woman s right to an abortion. In the past, he has said such issues are not  so important  compared to the fight against corruption. He has also proposed putting same-sex marriage and abortion to a popular vote, which rankles activists like Castaneda. Asked if she might still support Lopez Obrador, Castaneda was quick to answer.  Right now, I haven t made a decision.  "'''
    # word_freq_counter(news_context)

    label_balance_checker()