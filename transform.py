import openai
from retry import retry

api_key="YOUR_API_KEY"


@retry(Exception, tries=5, delay=1, backoff=2, max_delay=120)
def action_extraction(text, char_list):
    
    openai.api_key=api_key
    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": 
                f'''
                    **instruction**
                    In the "text", the character names are enclosed in "[" and "]". 
                    From the text, the list of characters is {char_list}. 
                    Extract all the actions of each character name. 
                    If there is no action, print "no action".
                    
                    **Definition of Action**
                    Social norm is a guideline for social conduct generally observed by most people in everyday situations.
                    Action is an action by the actor that fulfills the intention and observes or diverges from the social norm.

                    **output format**
                    The output format should be as follows: 
                    <start>\n
                    [character name]: "[character name] action sentence.",\n 
                    [character name]: "[character name] action sentence.",\n
                    [character name]: "[character name] action sentence.",\n
                    ...\n
                    <end>
                '''
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"text: {text}"
            }
        ]
        }
    ],
    temperature=0.01,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    output_text = response["choices"][0]["message"]["content"]
    return output_text


@retry(Exception, tries=5, delay=1, backoff=2, max_delay=120)
def context_extraction(char_name, action, segment):
    
    openai.api_key=api_key
    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": 
                f'''
                    **instruction**
                    Your work is to extract three sentences in the "Segment", corresponding Situation, Intention, and Consequence related to the {char_name}'s given "Action".
                    If any of the categories are not mentioned in the "Segment", please generate "not exist" for that category. \
                    
                    **Definition of Situation**
                    Situation is a setting of the story that introduces story participants and describes their environment.
                    
                    **Definition of Intention**
                    Intention is a Reasonable goal that the story participants (the actor), wants to fulfill.
                    
                    **Definitio of Consequence**
                    Consequence is a possible effect of the action on the actor's environment.

                    **output format**
                    The output format should be as follows: 
                    <start>\n
                    [Situation]: situation sentence.\n 
                    [Intention]: intention sentence.\n
                    [Consequence]: consequence sentence.\n
                    ...\n
                    <end>
                '''
            }
        ]
        },
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f'''
                    Segment: {segment}\n
                    Action: {action}\n
                    Action character: {char_name}
                '''
            }
        ]
        }
    ],
    temperature=0.01,
    max_tokens=512,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    output_text = response["choices"][0]["message"]["content"]
    return output_text