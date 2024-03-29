import os
import random
import json
import csv
import sys
import os
import logging
import xml.etree.ElementTree as ET
from utils import detokenizer


'''

In general, examples should be of the form:

{
'options': [opt_1, opt_2, ..., opt_m]
'label': l  # index of correct option
}

opt_i is an option of the form:

{
'premise': premise # the question premise (string)
'hypothesis': h # hypothesis answer (str) we calculate conditional P(hypothesis|premise)
'unc_presmise': up # the premise for calculating uncond likelihood (str)
'unc_hypothesis': uh # the hypothesis used for calculating uncond likelihood P(hypothesis) 
                     # this will often just be hypothesis but may differ slightly for format

}

'''


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def proc_question(s):
    s = s[0].upper() + s[1:]
    s = s.replace(' i ', ' I ')
    s = s + '?'
    return s

def load_examples_cqa(path, return_tuple=False):

    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            ## use the '?' as a bridge
            if not premise[-1] in '?.!':
                print(premise)
            else:
                premise = premise[:-1] ## trim the punctuation, will add a question mark
                
                
            if return_tuple:
                options = [ '? the answer is: "{}"'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [(premise,opt) for opt in options], 
                  'label':label}]
            else:
                options = [ '? {}'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [{'premise':premise + '? the answer is:' ,
                                          'hypothesis': ' "{}"'.format(c['text'].lower()),
                                           'uncond_premise': ' the answer is:',
                                           'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                          'label':label}]
    return examples

def load_examples_cqa_with_nle(path, return_tuple=False):
    # load data
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append(d)
    return examples

def load_examples_copa_prefix(path, return_tuple = False, prefix=''):
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
            
        # legacy, using tuples
        if return_tuple:
            examples_copa  += [{'options': [(' '+p[:-1] ,bridge + a1),
                                                (' '+p[:-1] , bridge + a2)], 
                      'label':int(value)-1, 'asks-for': asks_for, 'bridge':bridge}]
        else:
            examples_copa  += [{'options': [{'premise':prefix + ' '+p[:-1] + bridge,
                                             'hypothesis': ' '+ a1,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a1},
                                           {'premise':prefix + ' '+p[:-1] + bridge,
                                             'hypothesis': ' '+a2,
                                             'uncond_premise':bridge,
                                             'uncond_hypothesis':' '+a2}], 
                      'label':int(value)-1}]
    return examples_copa

def load_examples_obqa_prefix(path, prefix=""):
    with open(path) as lines:
        idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D' }
        abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }

        examples = []
        for line in lines:
            j = json.loads(line)
            d = {}

            label = j['answerKey']
            correct_hypothesis = abc2idx[label]
            q = j['question']
            stem = q['stem']
            choices = q['choices']
            hypotheses = []
            for idx, choice in enumerate(choices):
                text = choice['text']
                label = choice['label']
                assert(abc2idx[label] == idx)
                hypotheses.append(text)

            d['premise'] = prefix + stem
            d['hypotheses'] = hypotheses
            d['correct_hypothesis'] = correct_hypothesis

            d['stem'] = stem
            d['answers'] = choices
            d['label'] = label





            premise = d['premise']
            options = []
            for h in d['hypotheses']:
                o = {}
                h = ' ' + h
                o['premise'] = premise
                o['hypothesis'] = h
                o['uncond_premise'] = ' the answer is:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = d['correct_hypothesis']
            examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_cqa_prefix(path, return_tuple=False, prefix=''):

    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            ## use the '?' as a bridge
            if not premise[-1] in '?.!':
                print(premise)
            else:
                premise = premise[:-1] ## trim the punctuation, will add a question mark
                
                
            if return_tuple:
                options = [ '? the answer is: "{}"'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [(premise,opt) for opt in options], 
                  'label':label}]
            else:
                options = [ '? {}'.format(c['text'].lower()) for c in d['question']['choices']]
                examples += [{'options': [{'premise':prefix + premise + '? the answer is:' ,
                                          'hypothesis': ' "{}"'.format(c['text'].lower()),
                                           'uncond_premise': ' the answer is:',
                                           'uncond_hypothesis': ' "{}"'.format(c['text'].lower())} for c in d['question']['choices']], 
                          'label':label}]
    return examples

def load_examples_hellaswag_prefix(path, prefix=""):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]
    examples = []
    for d in data:
        premise = d["ctx"].strip()
        last_space_index = premise.rfind(' ')
        uncond_premise = premise[last_space_index:]

        options = []
        for hypothesis in d['endings']:
            o = { 'premise' : prefix + premise, 'uncond_premise' : uncond_premise } 
            o['hypothesis'] = ' ' + hypothesis
            o['uncond_hypothesis'] = ' ' + hypothesis
            options.append(o)
        label = d['label']
        examples.append( { 'options' : options, 'label' : label } )
    return examples

def load_examples_storycloze_prefix(path, return_tuple = False, prefix=""):
    data = []
    with open(path) as fp:
        reader = csv.DictReader(fp, delimiter = "\t")
        for row in reader:
            d = {}
            premise = f'{prefix}{row["InputSentence1"]}'
            premise = f'{premise} {row["InputSentence2"]}'
            premise = f'{premise} {row["InputSentence3"]}'
            premise = f'{premise} {row["InputSentence4"]}'
            d['premise'] = premise
            hypotheses = [ row['RandomFifthSentenceQuiz1'], row['RandomFifthSentenceQuiz2'] ]
            d['hypotheses'] =  hypotheses
            correct_hypothesis = int(row['AnswerRightEnding']) - 1
            d['correct_hypothesis'] = correct_hypothesis
            d['id'] = row['InputStoryid']
            data.append(d)
    examples = []
    for d in data:
        end = '.'
        # take the punctuation from the end of the story as a prefix to 
        # the last sentence, so that we have something to condition on
        # for P(final_sentence)
        if d['premise'][-1] in '!.':
            end = d['premise'][-1] 
            d['premise'] = d['premise'][:-1]
            
        if return_tuple:
            examples += [{'options':[(d['premise'],end +' ' +h) for h in d['hypotheses']],
                        'label':d['correct_hypothesis']}]
        else:
            examples += [{'options':[{'premise':d['premise'] + end,
                                      'hypothesis':  ' ' +h,
                                      'uncond_premise': ' The story continues:' ,
                                      'uncond_hypothesis':  ' ' + h }for h in d['hypotheses']],
                        'label':d['correct_hypothesis']}]
    return examples

def load_examples_boolq(path, **kwargs):
    data = []
    with open(path) as f:
        for line in f:
            data += [json.loads(line)]

    examples = []
    for d in data:
        options = []
        p = f' title: { d["title"]}\n question: {proc_question(d["question"])}\n answer:'
        if kwargs["multiple_target_words"] == True:
            # construct manual label words
            hypotheses = [[' yes', ' true'], [' no', ' false']]
        else:
            # hypotheses = [' yes', ' no']
            hypotheses = [' true', ' false']
        for h in hypotheses:
            o = {}
            o['premise'] = p
            o['hypothesis'] = h
            # o['uncond_premise'] = ' yes or no?\n answer:'
            o['uncond_premise'] = '\n answer:'
            o['uncond_hypothesis'] = h
            options.append(o)
        label = 1 if not d['answer'] else 0 #.strip().lower() == 'false' else 1
        examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_cqa_mcp(path, **kwargs):
    cond_mcp, uncond_mcp = kwargs['cond_mcp'], kwargs['uncond_mcp']
    if kwargs['domain_cond'] == "":
        domain_cond = " the answer is:"
    else:
        domain_cond = kwargs['domain_cond']
    
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            label = ['A','B','C','D','E'].index(d['answerKey'])
            premise = ' ' +d['question']['stem']
            ## use the '?' as a bridge
            if not premise[-1] in '?.!':
                print(premise)
            else:
                premise = premise[:-1] ## trim the punctuation, will add a question mark
            
            options = [c['text'].lower() for c in d['question']['choices']]
            if kwargs['with_symbol'] == True:
                options = [f"{char}. {option}" for char, option in zip(['A','B','C','D','E'], options)]
                examples += [{'options': [{'premise': cond_mcp.replace("[answers]", str(options)) + premise.lower() + '?' + domain_cond ,
                                'hypothesis': ' "{}"'.format(char),
                                'uncond_premise': uncond_mcp.replace("[answers]", str(options)) + domain_cond,
                                'uncond_hypothesis': ' "{}"'.format(char)} for char, option in zip(['A','B','C','D','E'], options)], 
                                'label':label}]
            else:
                examples += [{'options': [{'premise': cond_mcp.replace("[answers]", str(options)) + premise.lower() + '?' + domain_cond ,
                                      'hypothesis': ' "{}"'.format(option),
                                       'uncond_premise': uncond_mcp.replace("[answers]", str(options)) + domain_cond,
                                       'uncond_hypothesis': ' "{}"'.format(option)} for option in options], 
                      'label':label}]
    return examples

def load_examples_copa_mcp(path, **kwargs):
    cond_mcp, uncond_mcp, domain_cond = kwargs['cond_mcp'], kwargs['uncond_mcp'], kwargs['domain_cond']
    
    root = ET.parse(path).getroot()
    examples_copa = []
    for type_tag in root.findall('item'):
        # xml stuff
        value = type_tag.get('most-plausible-alternative')
        asks_for = type_tag.get('asks-for')
        children = list(type_tag)
        # get the texts
        p = children[0].text
        a1 = children[1].text[:1].lower() +  children[1].text[1:]
        a2 = children[2].text[:1].lower() +  children[2].text[1:]
        options = [a1, a2]
        if asks_for =='effect':
            bridge = ' so'
        elif asks_for =='cause':
            bridge = ' because'
        else: 
            assert(False)
            
        examples_copa  += [{'options': [{'premise':' '+ cond_mcp.replace("[answers]", str(options)) + p[:-1] + bridge + '.' + domain_cond,
                                         'hypothesis': ' '+ a1,
                                         'uncond_premise':uncond_mcp.replace("[answers]", str(options)) + bridge + '.' + domain_cond,
                                         'uncond_hypothesis':' '+a1},
                                       {'premise':' '+ cond_mcp.replace("[answers]", str(options)) + p[:-1] + bridge + '.' + domain_cond,
                                         'hypothesis': ' '+a2,
                                         'uncond_premise':uncond_mcp.replace("[answers]", str(options)) + bridge + '.' + domain_cond,
                                         'uncond_hypothesis':' '+a2}], 
                  'label':int(value)-1}]
    return examples_copa

'''

This loads COPA, putting hypothesis before the premise

(so forward LM score is PMI)

'''

def load_examples_obqa_mcp(path, **kwargs):
    cond_mcp, uncond_mcp, domain_cond = kwargs['cond_mcp'], kwargs['uncond_mcp'], kwargs['domain_cond']
    with open(path) as lines:
        idx2abc = { 0 : 'A', 1 : 'B', 2 : 'C', 3 : 'D' }
        abc2idx = { 'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3 }

        examples = []
        for line in lines:
            j = json.loads(line)
            d = {}

            label = j['answerKey']
            correct_hypothesis = abc2idx[label]
            q = j['question']
            stem = q['stem']
            choices = q['choices']
            hypotheses = []
            for idx, choice in enumerate(choices):
                text = choice['text']
                label = choice['label']
                assert(abc2idx[label] == idx)
                hypotheses.append(text)

            d['premise'] = stem
            d['hypotheses'] = hypotheses
            d['correct_hypothesis'] = correct_hypothesis

            d['stem'] = stem
            d['answers'] = choices
            d['label'] = label

            premise = d['premise']
            options = []
            # anaswers = d['hypotheses']
            for h in d['hypotheses']:
                o = {}
                h = ' ' + h
                o['premise'] = cond_mcp.replace("[answers]", str(hypotheses)) + premise + '.' + domain_cond
                o['hypothesis'] = h
                o['uncond_premise'] = uncond_mcp.replace("[answers]", str(hypotheses)) + domain_cond # + ' the answer is:'
                o['uncond_hypothesis'] = h
                options.append(o)
            label = d['correct_hypothesis']
            examples.append({'options' : options, 'label' : label })
    return examples

def load_examples_piqa_mcp(qa_path, label_path, **kwargs):
    cond_mcp, uncond_mcp, domain_cond = kwargs['cond_mcp'], kwargs['uncond_mcp'], kwargs['domain_cond'] #" The answer is: "
    if domain_cond == '':
        domain_cond = " The answer is: "

    examples = []
    with open(qa_path) as lines, open(label_path) as labels:
        for line, label in zip(lines, labels):
            example = {}
            example['label'] = int(label[0])

            line = json.loads(line)
            options = []
            all_sol= [line['sol1'], line['sol2']]
            for key in ['sol1', 'sol2']:
                option = {}
                option['hypothesis'] = line[key]
                option['premise'] = cond_mcp.replace("[answers]", str(all_sol)) + line['goal'] + domain_cond
                option['uncond_hypothesis'] = line[key]
                option['uncond_premise'] = uncond_mcp.replace("[answers]", str(all_sol)) + domain_cond
                options.append(option)
            example['options'] = options
            examples.append(example)
    
    return examples

def load_examples_siqa_mcp(qa_path, label_path, **kwargs):
    cond_mcp, uncond_mcp, domain_cond = kwargs['cond_mcp'], kwargs['uncond_mcp'], kwargs['domain_cond'] #" The answer is: "
    
    if domain_cond == '':
        domain_cond = " The answer is: "

    examples = []
    with open(qa_path) as lines, open(label_path) as labels:
        for line, label in zip(lines, labels):
            example = {}
            example['label'] = int(label[0]) - 1 # This took me at least an hour to figure out.

            line = json.loads(line)
            options = []
            
            if kwargs['with_symbol'] == True:
                all_sol = [f"A. {line['answerA']}", f"B. {line['answerB']}", f"C. {line['answerC']}"]
            else:
                all_sol = [line['answerA'], line['answerB'], line['answerC']]
            
            for key, char in zip(['answerA', 'answerB', 'answerC'], ['A', 'B', 'C']):
                option = {}
                
                if kwargs["with_symbol"] == True:
                    option['hypothesis'] = f"{char}"
                    option['uncond_hypothesis'] = f"{char}"
                else:
                    option['hypothesis'] = line[key]
                    option['uncond_hypothesis'] = line[key]

                option['premise'] = cond_mcp.replace("[answers]", str(all_sol)) + line['context'] + ' ' + line['question'] + domain_cond
                option['uncond_premise'] = uncond_mcp.replace("[answers]", str(all_sol)) + domain_cond
                options.append(option)
            example['options'] = options
            examples.append(example)
    
    return examples