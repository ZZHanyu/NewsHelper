
from openai import OpenAI
import httpx
import json

import time 


def get_response(prompt):
  client = OpenAI(
      base_url="https://api.xty.app/v1", 
      api_key="sk-POqdjx08gDq6DuTcF5F75c2e4aC4411b909287B7F8A49a6a",
      http_client=httpx.Client(
        base_url="https://api.xty.app/v1",
        follow_redirects=True,
      ),
    )

    # completion = client.chat.completions.create(
    #   model="gpt-3.5-turbo",
    #   messages=[
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello!"}
    #   ]
    # )

  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        prompt[0]
    ]
  )

  # print(completion)
  print(completion.choices[0].message.content)

  return completion





def get_prompt_test():
    # prompt
    news = '''Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.  Loretta Lynch aka Eric Holder in a skir.'''
    news2 = '''GOP Senator Just Smacked Down The Most Punchable Alt-Right Nazi On The Internet,"The most punchable Alt-Right Nazi on the internet just got a thorough beatdown from Sen. Ben Sasse (R-Neb.) on Twitter during an epic tweetstorm. Richard Spencer, the Alt-Right leader who has become a human punching bag, just got the racism smacked out of him by the Republican Senator on Thursday after the white nationalist tweeted that only goober conservatives  blame Russia for  racial divisions  in the United States. Spencer was responding to a tweet Sasse sent out on Wednesday.Sen. Ben Sasse had shared an article regarding Sen. James Lankford (R-Okla.), who explained that Russian internet trolls helped fuel divisions in a controversy which Donald Trump ignited over NFL athletes who choose to kneel rather than stand during the national anthem in protest of racial inequality and police brutality.No one loves American-vs-American fighting more than Putin. His intel agencies stoke both sides of every divide.https://t.co/H6BwjHzokH  Ben Sasse (@BenSasse) September 28, 2017Spencer responded by writing,  In the minds of goober conservatives, the Russians are to blame for racial divisions. In the minds of goober conservatives, the Russians are to blame for racial divisions. https://t.co/CzpGfL6u4M  Richard  ?Spencer (@RichardBSpencer) September 28, 2017Sasse tore into Spencer, calling him a  clown  and one of the  brown-shirt-pajama-boy Nazis. 1/Oh let goobers & nongoobers agree on this: Racists like you are to blame. But Putin's agencies also love using you as their divisive tool https://t.co/DaD4XaNvI5  Ben Sasse (@BenSasse) September 28, 20172/Don t get me wrong: we ll always have brown-shirt-pajama-boy Nazis like you & your lonely pals stoking division. But here s America 101: https://t.co/SboVZmOuu2  Ben Sasse (@BenSasse) September 28, 20173/You don t get America. You said:  You do not have some human right, some abstract thing given to you by God or something like that.  https://t.co/ScXDGFcbGp  Ben Sasse (@BenSasse) September 28, 20174/Actually, that's exactly what America declares we do have: People are the image-bearers of God, created with dignity& inalienable rights. https://t.co/d4orBrHJMw  Ben Sasse (@BenSasse) September 28, 20175/Sadly, you don't understand human dignity. A person's skin, ancestry, and bank balance have nothing to do with their intrinsic value. https://t.co/5JsyVAKQRL  Ben Sasse (@BenSasse) September 28, 20176/This declaration of universal dignity is what America is about. Madison called our Constitution ""the greatest reflection on human nature"" https://t.co/NQluVs1KvA  Ben Sasse (@BenSasse) September 28, 20177/You talk about culture but don't know squat about western heritage which sees people not as tribes but as individuals of limitless worth https://t.co/VKNvDUXLtT  Ben Sasse (@BenSasse) September 28, 20178/The celebration of universal dignity IS our culture, & it rejects your ""white culture"" crybaby politics. It rejects all identity politics https://t.co/Adlj9AvNPR  Ben Sasse (@BenSasse) September 28, 20179/Sometime after moving back into your parents' basement, you knock-off Nazis fell in love with reheated 20th century will-to-power garbage https://t.co/XDTeATVGSe  Ben Sasse (@BenSasse) September 28, 201710/Your ""ideas"" aren't just hateful, un-American poison they're also just so dang boring. The future doesn't belong to your stupid memes. https://t.co/bNHSlf1uOx  Ben Sasse (@BenSasse) September 28, 201711/11Get a real job, Clown. Find an actual neighbor to serve. You'll be happier.Have a nice day. https://t.co/ToREd7VwDM  Ben Sasse (@BenSasse) September 28, 2017Jake Tapper weighed in to say,  Wherein a piece of garbage is thrown into a receptacle with grace and artistry. Wherein a piece of garbage is thrown into a receptacle with grace and artistry https://t.co/L09bBy8gHh  Jake Tapper (@jaketapper) September 28, 2017This is how Donald Trump should have responded to the violent rally in Charlottesville, Virginia, which, by the way, was organized by Richard Spencer.Spencer previously said that he felt  proud  after Trump blamed  many sides  for the violence at the hate-rally which resulted in the murder of Heather Heyer, and left at least 19 others injured. Spencer is one of Trump s  very fine people  who just got smacked down on Twitter by a Republican.You don t have to like his politics to admire that Sasse was among the Republicans who joined civil rights leaders and Democrats who reacted angrily when Trump said that he condemned  this egregious display of hatred, bigotry, and violence on many sides   on many sides. Of course, if Sasse would call out Roy Moore over his bigoted remarks, that would be nice to see, too.Photo: Chip Somodevilla via Getty images.'''
    news3 = '''May Brexit offer would hurt, cost EU citizens - EU parliament","BRUSSELS (Reuters) - British Prime Minister Theresa May s offer of  settled status  for EU residents is flawed and will leave them with fewer rights after Brexit, the European Parliament s Brexit coordinator said on Tuesday. A family of five could face a bill of 360 pounds to acquire the new status, Guy Verhofstadt told May s Brexit Secretary David Davis in a letter seen by Reuters    a very significant amount for a family on low income . Listing three other concerns for the EU legislature, which must approve any treaty on the March 2019 exit, Verhofstadt told Davis:  Under your proposals, EU citizens will definitely notice a deterioration of their status as a result of Brexit. And the Parliament s aim all along has been that EU citizens, and UK citizens in the EU-27, should notice no difference.  Verhofstadt, a former Belgian prime minister, wrote in response to Davis, who had written to him after Parliament complained last week that there remained  major issues  to be settled on the rights of the 3 million EU citizens in Britain. On Tuesday, he told reporters that Parliament was determined that expatriates should not become  victims of Brexit . May had unveiled more details last week of a system aimed at giving people already in Britain a quick and cheap way of asserting their rights to stay there indefinitely. The issue, along with how much Britain owes and the new EU-UK border across Ireland, is one on which the EU wants an outline agreement before opening talks on the future of trade. Verhofstadt said lawmakers were not dismissing British efforts to streamline applications but saw flaws in the nature of  settled status  itself. As well as the cost, which is similar to that of acquiring a British passport, he cited three others: - Europeans should simply  declare  a whole household resident, without needing an  application  process; the burden of proof should be on the British authorities to deny them rights. - more stringent conditions on criminal records could mean some EU residents, including some now with permanent resident status, being deported for failing to gain  settled status . - EU residents would lose some rights to bring relatives to Britain as the new status would give them the same rights as British people, who now have fewer rights than EU citizens.'''
    news4 = '''Three Arrested for Starting Massive Fire on Atlanta Interstate Bridge,"Three people were arrested Friday in connection with a massive fire under a bridge on Interstate 85 in Atlanta that caused part of the bridge to collapse. [Deputy Insurance Commissioner Jay Florence said Basil Eleby, Sophia Bruner, and Barry Thomas were the three suspects arrested, the Daily Mail reported.  Eleby, whom Florence believes started the fire, faces a charge of criminal damage to property. Bruner and Thomas each face charges of criminal trespass. “We believe they were together when the fire was set and Eleby is the one who set the fire,” Florence told the Atlanta  . Florence did not release details of how or why the fire was started, saying that those details would be released as the investigation continues. He said the suspects may have used “available materials” at the site and added that all three suspects may have been homeless. Crews worked to restore the bridge Thursday after the blaze caused it to collapse during rush hour. Nobody was injured despite the large clouds of smoke and flames that filled the air. Officials say reconstruction of the damaged section of the bridge could take several months, causing the 225, 000 commuters who use that section of the highway daily a traffic nightmare.'''


    instruction = '''Task: Classify what the labels of the news described in <News Context> below. 
    Requirement 1. No explaination or think of chain rquired. Just give the words that describe the class (e.g., Politics,Sports, Entertainment);
    Requirement 2. The word or words needs to summarize the theme well;
    '''
    # one-shot prompt:

    # example = f''' 
    # <system>: - Instruction: "{instruction}".
    #     - News Context:: "{news}".
    # <bot>: Answer: label is "1. Politics, 2. Laws" .
    # '''

    messages = [
        {
            "role": "system",
            "content": f'''- Instruction: "{instruction}". - News Context:: "{news}"; - Your classification result = ".
            ''',
        },

        {"role": "assistant", "content": "Answer: The classification result =  '1. Politics, 2. Laws'."},

        {"role": "user", "content": f"- Instruction: {instruction} \n-News Context {news2}\n" },
    ]

    # input_prompt = f'''
    # <system>: Below is an instruction that describes a task. Write a answer that appropriately completes the instruction, answer format need same with example: 
    # Example:<{example}>.
    # <system>: - Instruction: "{instruction}".
    #     - News Context: "{news4}".
    # <bot>: Answer: label is 
    #     '''

    return messages





def get_prompt(news):
    instruction = '''Task: Classify what the labels of the news described in <News Context> below. 
      Requirement 1. No explaination or think of chain rquired. Just give the words that describe the class (e.g., Politics,Sports, Entertainment);
      Requirement 2. The word or words needs to summarize the theme well;
      '''
    news_example = '''Now, most of the demonstrators gathered last night were exercising their constitutional and protected right to peaceful protest in order to raise issues and create change.  Loretta Lynch aka Eric Holder in a skir.'''

    messages = [
        {
            "role": "system",
            "content": f'''- Instruction: "{instruction}". - News Context:: "{news_example}"; - Your classification result = ".
            ''',
        },

        {"role": "assistant", "content": "Answer: The classification result =  '1. Laws, 2. dissent'."},

        {"role": "user", "content": f"- Instruction: {instruction} \n-News Context {news}\n" },
    ]

    return messages




# test port
if __name__ == "__main__":
    # message = get_prompt()
    # get_response(message)

    with open('/root/autodl-tmp/NewsHelper/tp_result.json', 'r') as f:
      tp_result = json.load(f)
      print(tp_result['3'][1])

      prompt = get_prompt(tp_result['3'][1])
      result = get_response(prompt)
      print(result)

      time.sleep(2)



