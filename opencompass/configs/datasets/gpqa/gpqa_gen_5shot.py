from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GPQADataset
from opencompass.utils.text_postprocessors import first_capital_postprocess

hint = f'For the multiple choice question below, please provide the correct answer option directly.'

gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template='Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {answer}\n',
    ),
    prompt_template=dict(
        type=PromptTemplate,
        template=f'{hint}\n</E>Question: {{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer:',
        ice_token='</E>'
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
    inferencer=dict(type=GenInferencer),
)

gpqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess),
)

gpqa_datasets = []
gpqa_subsets = {
    # 'extended': 'gpqa_extended.csv',
    # 'main': 'gpqa_main.csv',
    'diamond': 'gpqa_diamond.csv'
}

for split in list(gpqa_subsets.keys()):
    gpqa_datasets.append(
        dict(
            abbr='GPQA_' + split + '_5shot',
            type=GPQADataset,
            path='./data/gpqa/',
            name=gpqa_subsets[split],
            reader_cfg=gpqa_reader_cfg,
            infer_cfg=gpqa_infer_cfg,
            eval_cfg=gpqa_eval_cfg)
    )
