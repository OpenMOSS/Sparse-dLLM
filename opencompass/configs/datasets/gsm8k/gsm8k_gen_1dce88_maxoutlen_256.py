from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=
        '''Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: {question}\nAnswer:'''),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=256))

gsm8k_eval_cfg = dict(
    evaluator=dict(type=Gsm8kEvaluator),
    pred_postprocessor=dict(type=gsm8k_postprocess),
    dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='opencompass/gsm8k',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
