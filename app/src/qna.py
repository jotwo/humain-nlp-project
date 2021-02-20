import sys
import torch 
from transformers import BertForQuestionAnswering, BertTokenizer
import pandas as pd

def qa(usecase_indication):
    # there is 2 options 1-load Bert from online  2-load-save then load from local folder 
    # I commented following 2 lines
    # model = BertForQuestionAnswering.from_pretrained  ('bert-large-uncased-whole-word-masking-finetuned-squad') # 1-load Bert from online
    # model.save_pretrained("./BertLSquad") # to save the model for regualar local use 
    model = BertForQuestionAnswering.from_pretrained('./BertLSquad') #2- open saved model
    print('QnA model loaded')
    sys.stdout.flush()
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    df1 = pd.DataFrame(columns=['industry','function', 'usecase', 'paragraph_id'])

    def answer_question(tokenizer, question, answer_text):
        '''
        Takes a `question` string and an `answer_text` string (which contains the
        answer), and identifies the words within the `answer_text` that are the
        answer. Prints them out.
        '''
        # ======== Tokenize ========
        # Apply the tokenizer to the input text, treating them as a text-pair.
        input_ids = tokenizer.encode(question, answer_text)

        # Report how long the input sequence is.
        # print('Query has {:,} tokens.\n'.format(len(input_ids)))

        # ======== Set Segment IDs ========
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input_ids)

        # ======== Evaluate ========
        # Run our example through the model.
        outputs = model(torch.tensor([input_ids]), # The tokens representing our input text.
                        token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                        return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # ======== Reconstruct Answer ========
        # Find the tokens with the highest `start` and `end` scores.
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Get the string versions of the input tokens.
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Start with the first token.
        answer = tokens[answer_start]

        # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):
            
            # If it's a subword token, then recombine it with the previous token.
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            
            # Otherwise, add a space then the token.
            else:
                answer += ' ' + tokens[i]

        # print('Answer: "' + answer + '"')
        return answer

    from pickle import dump, load
    # with open('usecase_indicator.pkl','rb') as f:
    #     usecase_indication = load(f)

    philippe = usecase_indication['paragraph_id'].values

    ##### we can ask one question per one paragraph or per multi paragraphs
    paragraphs=len(philippe)   
    counter=0 
    questions=["Which industries?" , "What is main idea?", "what kind of application?"]
    print("process is started ---------------------------------------")
    sys.stdout.flush()
    all_paragraphs = usecase_indication["paragraph"]
    for i in range(counter,len(all_paragraphs)):        
        answer_text=all_paragraphs[i]
        if paragraphs != counter :        
            for question in questions:
                answer = answer_question(tokenizer, question, answer_text)            
                if question == "Which industries?":
                    industry = answer
                elif question == "What is main idea?":
                    use_case = answer
                else:
                    ai_function = answer            
            df1 = df1.append({'industry': industry,'function':ai_function, 'usecase':use_case, 'paragraph_id':philippe[i]}, ignore_index=True) 
            print(counter, end = ' ')
            sys.stdout.flush()
            counter += 1
        else:
            break
    # df1.to_csv('QnA.csv',index=False)
    df1['doc_name']=usecase_indication['doc_id']
    df1['sentence']=usecase_indication['sentence']
    df1['usecase_score']=usecase_indication['usecase_score']
    df1['paragraph']=usecase_indication['paragraph']
    with open('result.pkl','wb') as f:
        dump(df1, f)
    
    return df1