import json

import datasets
from datasets import load_dataset

z_scrolls_datasets = ['summ_screen_fd', 'qasper', 'qmsum', 'narrative_qa', 'gov_report', 'quality', 'squality', 'musique', 'space_digest', 'book_sum_sort']
for tmp_dataset in z_scrolls_datasets:
    dataset = datasets.load_from_disk(f"/data0/maqi/KGLQA-data/datasets/zero_scrolls_datasets/{tmp_dataset}")
    tmp_pred_data = {}
    for item in dataset['test']:
        tmp_pred_data[item['id']] = 'A'
    tmp_sava_path = f'/data0/maqi/KGLQA-data/datasets/QuALITY/predictions/zero_scrolls/empty/{tmp_dataset}_pred.json'
    json.dump(tmp_pred_data, open(tmp_sava_path, 'w'))
    print(f'Finish {tmp_dataset}')
