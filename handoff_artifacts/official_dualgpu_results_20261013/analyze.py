"""Analyze complete PT/LP logs using only the Python standard library."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

FOLDER=Path(__file__).resolve().parent


def load(name, count):
    path=FOLDER/(name+'.jsonl')
    rows=[]
    for number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():
            continue
        row=json.loads(line)
        for key,value in row.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and not math.isfinite(value):
                raise ValueError('Non-finite metric at {}:{}:{}'.format(path,number,key))
        rows.append(row)
    if [row['epoch'] for row in rows]!=list(range(count)):
        raise ValueError('Missing, duplicate or unordered epochs: '+str(path))
    return rows


def lp_stats(rows):
    best=max(rows,key=lambda row:row['test_acc1'])
    tail=[row['test_acc1'] for row in rows[-20:]]
    return {'epochs':len(rows),'best_lp_acc':best['test_acc1'],'best_epoch':best['epoch'],
            'last_lp_acc':rows[-1]['test_acc1'],'last20_mean':statistics.mean(tail),
            'last20_population_std':statistics.pstdev(tail),
            'best_acc5':max(row['test_acc5'] for row in rows),
            'last_test_loss':rows[-1]['test_loss'],
            'n_parameters_unique':sorted(set(row['n_parameters'] for row in rows)),
            'epochs_at_lr_zero':[row['epoch'] for row in rows if row['train_lr']==0]}


pt=load('pt',400)
lp1=load('lp1',100)
lp2=load('lp2',100)
row_diff=[one['epoch'] for one,two in zip(lp1,lp2) if one!=two]
column_diff={key:max(abs(one[key]-two[key]) for one,two in zip(lp1,lp2))
             for key in lp1[0] if type(lp1[0][key]) in (int,float)}
pt_tail=[row['train_loss'] for row in pt[-20:]]
pt_stats={'epochs':len(pt),'first_loss':pt[0]['train_loss'],'last_loss':pt[-1]['train_loss'],
          'last20_loss_mean':statistics.mean(pt_tail),'last20_loss_population_std':statistics.pstdev(pt_tail),
          'last50_endpoint_loss_decrease_percent':100*(pt[350]['train_loss']-pt[-1]['train_loss'])/pt[350]['train_loss'],
          'max_epoch_mean_lr':max(row['train_lr'] for row in pt),
          'last_lr':pt[-1]['train_lr'],
          'epochs_at_min_lr_1e5':[row['epoch'] for row in pt if math.isclose(row['train_lr'],1e-5,rel_tol=1e-10)]}
result={'experiment_id':'T20','date_received':'2026-10-13',
        'source':'three user attachments: one PT and two user-confirmed same-seed LP executions',
        'pt':pt_stats,'lp_runs':{'lp1':lp_stats(lp1),'lp2':lp_stats(lp2)},
        'repeat_comparison':{'bytes_identical':(FOLDER/'lp1.jsonl').read_bytes()==(FOLDER/'lp2.jsonl').read_bytes(),
                             'all_fields_identical':lp1==lp2,'differing_epochs':row_diff,'max_abs_difference_by_field':column_diff},
        'user_confirmed_lp_identity':{'lp1_seed':0,'lp2_seed':0,'lp1_output_directory_name':'lp','lp2_output_directory_name':'lp1',
                                     'same_algorithm_arguments':True,'same_pretrained_encoder':True},
        'protocol_from_user_description_and_saved_command':{'source':'isolated official 692888f plus inf import compatibility and current data paths',
                                                           'native':True,'normalization':'A','decoder_depth':5,'pt_epochs':400,
                                                           'pt_world_size':2,'pt_per_gpu_batch':32,'pt_accum_iter':1,'pt_effective_batch':64,
                                                           'pt_seed_in_saved_command':0,'pt_lr':.001,'pt_min_lr':.00001,'pt_random_rot':False,
                                                           'lp_epochs':100,'lp_world_size':2,'lp_per_gpu_batch':64,'lp_accum_iter':1,'lp_effective_batch':128,
                                                           'lp_seed_user_confirmed':0,'lp_lr':.1,'lp_checkpoint_epoch_in_saved_command':399},
        'comparison':{'gap_to_user_reported_third_party_86_2_pp':86.2-lp_stats(lp1)['best_lp_acc'],
                      'approx_difference_from_T19_oral_85_89_pp':lp_stats(lp1)['best_lp_acc']-85.89},
        'log_integrity':{name:{'sha256':hashlib.sha256((FOLDER/(name+'.jsonl')).read_bytes()).hexdigest(),
                               'size_bytes':(FOLDER/(name+'.jsonl')).stat().st_size} for name in ('pt','lp1','lp2')},
        'limitations':['Full epoch metrics are provided and verified, but logs contain no Namespace/source/data/checkpoint SHA or environment versions.',
                       'Same-seed LP execution reproduction does not estimate across-seed LP variance or PT seed variance.',
                       'Byte-identical logs do not prove byte-identical checkpoints; checkpoints were not provided.',
                       'PT decoder/source/batch identity follows user description and prepared command; LR and epoch coverage are directly observed.']}
(FOLDER/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (FOLDER/'lp_repeat_comparison.csv').open('w',newline='',encoding='utf-8') as output:
    fields=['epoch','lp1_acc1','lp2_acc1','acc1_delta','lp1_train_loss','lp2_train_loss','train_loss_delta']
    writer=csv.DictWriter(output,fieldnames=fields)
    writer.writeheader()
    for one,two in zip(lp1,lp2):
        writer.writerow({'epoch':one['epoch'],'lp1_acc1':one['test_acc1'],'lp2_acc1':two['test_acc1'],
                         'acc1_delta':two['test_acc1']-one['test_acc1'],'lp1_train_loss':one['train_loss'],
                         'lp2_train_loss':two['train_loss'],'train_loss_delta':two['train_loss']-one['train_loss']})
print(json.dumps({'pt':pt_stats,'lp1':result['lp_runs']['lp1'],'lp2':result['lp_runs']['lp2'],
                  'repeat_comparison':result['repeat_comparison'],'comparison':result['comparison']},ensure_ascii=False,indent=2))
