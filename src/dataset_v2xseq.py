import sys
sys.path.insert(0, "/scratch/muminul951/v2x/V2X-Graph/required/DAIR-V2X-Seq/projects/TNT_plugin")
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# V2X-Seq uses 50 obs + 50 pred (confirmed from TNT_plugin horizon=50 and has_obss/has_preds both full 50)
HIDDEN_SIZE = 128
OBS_LEN = 50
PRED_LEN = 50

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3


def pad_vector(li):
    assert len(li) <= HIDDEN_SIZE
    return li + [0.0] * (HIDDEN_SIZE - len(li))


def build_agent_vectors(feats, has_obss):
    
    vectors = []
    polyline_spans = []
    agents = []

    for i in range(feats.shape[0]):
        traj_xy = feats[i, :OBS_LEN, :2]
        valid = has_obss[i, :OBS_LEN]

        start = len(vectors)
        agent_pts = []

        for t in range(1, OBS_LEN):
            if not valid[t] or not valid[t - 1]:
                continue
            pre_x, pre_y = traj_xy[t - 1, 0], traj_xy[t - 1, 1]
            x, y = traj_xy[t, 0], traj_xy[t, 1]
            is_target = 1 if i == 0 else 0
            is_other = 0 if i == 0 else 1
            vec = pad_vector([pre_x, pre_y, x, y, float(t), 0.0, float(is_target), float(is_other),
                              float(len(polyline_spans)), float(t)])
            vectors.append(vec)
            agent_pts.append([x, y])

        end = len(vectors)
        if end > start:
            polyline_spans.append([start, end])
            agents.append(np.array(agent_pts))

    return vectors, polyline_spans, agents


def build_lane_vectors(graph, offset):
    
    ctrs = graph['ctrs']         
    feats = graph['feats']      
    turn = graph['turn']         
    control = graph['control']   
    intersect = graph['intersect']  
    lane_idcs = graph['lane_idcs']  
    vectors = []
    polyline_spans = []
    polygons = []

    num_lanes = int(lane_idcs.max()) + 1

    for lane_id in range(num_lanes):
        mask = lane_idcs == lane_id
        nodes = np.where(mask)[0]
        if len(nodes) < 2:
            continue

        lane_ctrs = ctrs[mask]
        lane_feats = feats[mask]
        lane_turn = turn[mask]
        lane_ctrl = control[mask]
        lane_inter = intersect[mask]

        # Reconstruct polyline: each node center ± half the direction vector
        polyline = []
        for j in range(len(lane_ctrs)):
            pt = lane_ctrs[j] - lane_feats[j] * 0.5
            polyline.append(pt)
        polyline.append(lane_ctrs[-1] + lane_feats[-1] * 0.5)
        polyline = np.array(polyline)
        polygons.append(polyline)

        start = offset + len(vectors)
        point_pre = None

        for j, point in enumerate(polyline):
            if point_pre is None:
                point_pre = point
                point_pre_pre = point
                continue

            vec = np.zeros(HIDDEN_SIZE)
            vec[-1 - VECTOR_PRE_X] = point_pre[0]
            vec[-1 - VECTOR_PRE_Y] = point_pre[1]
            vec[-1 - VECTOR_X] = point[0]
            vec[-1 - VECTOR_Y] = point[1]
            vec[-5] = 1.0                      # is_lane
            vec[-6] = float(j)                 # step in polyline
            vec[-7] = float(len(polyline_spans))  # polyline index

            node_idx = min(j - 1, len(lane_ctrl) - 1)
            vec[-8] = float(lane_ctrl[node_idx])
            turn_dir = 1 if lane_turn[node_idx, 0] else (-1 if lane_turn[node_idx, 1] else 0)
            vec[-9] = float(turn_dir)
            vec[-10] = float(lane_inter[node_idx])

            vec[-17] = point_pre_pre[0]
            vec[-18] = point_pre_pre[1]

            vectors.append(vec.tolist())
            point_pre_pre = point_pre
            point_pre = point

        end = offset + len(vectors)
        if end > start:
            polyline_spans.append([start, end])

    return vectors, polyline_spans, polygons


def convert_sample(raw_data, args):
    feats = raw_data['feats'].values[0]        
    has_obss = raw_data['has_obss'].values[0]  
    gt_preds = raw_data['gt_preds'].values[0]  
    tar_candts = raw_data['tar_candts'].values[0]  
    gt_candts = raw_data['gt_candts'].values[0]   
    graph = raw_data['graph'].values[0]
    orig = raw_data['orig'].values[0]          
    theta = float(raw_data['theta'].values[0])
    seq_id = str(raw_data['seq_id'].values[0])
    city = raw_data['city'].values[0]

    agent_vecs, agent_spans, agents = build_agent_vectors(feats, has_obss)
    if len(agent_spans) == 0:
        return None

    map_start_polyline_idx = len(agent_spans)
    lane_vecs, lane_spans, polygons = build_lane_vectors(graph, offset=len(agent_vecs))

    all_vecs = agent_vecs + lane_vecs
    all_spans = agent_spans + lane_spans

    matrix = np.array(all_vecs, dtype=np.float32)
    labels = gt_preds[0] 
    goals_2D = tar_candts 
    point_label = labels[-1]
    goals_2D_labels = int(np.argmin(np.linalg.norm(goals_2D - point_label, axis=1)))

    # Closest lane polygon to final position (for lane scoring)
    stage_one_label = 0
    if len(polygons) > 0:
        min_dis = float('inf')
        for i, poly in enumerate(polygons):
            d = np.min(np.linalg.norm(poly - point_label, axis=1))
            if d < min_dis:
                min_dis = d
                stage_one_label = i
    # V2X-Seq trajectories are already in local agent-centric frame.
    # Setting cent to UTM origin (~417894, 4730251) causes to_origin_coordinate()
    # to produce predictions ~4.7M meters off.
    mapping = dict(
        file_name=seq_id + '.csv',
        city_name=city,
        cent_x=0.0,
        cent_y=0.0,
        angle=0.0,
        matrix=matrix,
        labels=labels.astype(np.float32),
        origin_labels=labels.astype(np.float32),
        polyline_spans=[slice(s[0], s[1]) for s in all_spans],
        labels_is_valid=np.ones(PRED_LEN, dtype=np.int64),
        eval_time=50,
        map_start_polyline_idx=map_start_polyline_idx,
        goals_2D=goals_2D.astype(np.float32),
        goals_2D_labels=goals_2D_labels,
        stage_one_label=stage_one_label,
        agents=agents,
        trajs=agents,
        polygons=polygons,
    )

    return mapping


def get_raw_dir(data_dir, is_train):
    subdir = 'train_intermediate' if is_train else 'val_intermediate'
    return os.path.join(data_dir, subdir, 'raw')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        self.args = args
        self.batch_size = batch_size
        self.ex_list = []

        is_train = not args.do_eval
        raw_dir = get_raw_dir(args.data_dir[0], is_train)
        assert os.path.exists(raw_dir), f"Raw dir not found: {raw_dir}"

        pkl_files = sorted([f for f in os.listdir(raw_dir) if f.endswith('.pkl')])
        assert len(pkl_files) > 0, f"No pkl files found in {raw_dir}"

        for fname in tqdm(pkl_files, desc=f"Loading V2X-Seq ({'train' if is_train else 'val'})"):
            raw_data = pd.read_pickle(os.path.join(raw_dir, fname))
            mapping = convert_sample(raw_data, args)
            if mapping is not None:
                self.ex_list.append(zlib.compress(pickle.dumps(mapping)))

        if to_screen:
            print(f"Loaded {len(self.ex_list)} samples from {raw_dir}")

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        return pickle.loads(zlib.decompress(self.ex_list[idx]))

def post_eval(args, file2pred, file2labels, DEs):
    import numpy as np
    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
    score_file += '.score'

    minADEs, minFDEs, MRs = [], [], []
    for seq_id in file2pred:
        preds = file2pred[seq_id]   
        gt = file2labels[seq_id]   
        fdes = [np.linalg.norm(p[-1] - gt[-1]) for p in preds]
        ades = [np.mean(np.linalg.norm(p - gt, axis=1)) for p in preds]
        minFDEs.append(min(fdes))
        minADEs.append(min(ades))
        MRs.append(1.0 if min(fdes) > 2.0 else 0.0)

    metric_results = {
        'minADE': np.mean(minADEs),
        'minFDE': np.mean(minFDEs),
        'MR': np.mean(MRs)
    }

    import utils
    utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)

    if DEs:
        DE = np.concatenate(DEs, axis=0)
        length = DE.shape[1]
        DE_score = [0, 0, 0, 0]
        for i in range(DE.shape[0]):
            DE_score[0] += DE[i].mean()
            for j in range(1, 4):
                index = round(float(length) * j / 3) - 1
                DE_score[j] += DE[i][index]
        for j in range(4):
            score = DE_score[j] / DE.shape[0]
            utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3',
                         score, type=score_file, to_screen=True, append_time=True)

    print(f"\n=== DenseTNT Eval Results ===")
    print(f"minADE: {metric_results['minADE']:.4f}")
    print(f"minFDE: {metric_results['minFDE']:.4f}")
    print(f"MR:     {metric_results['MR']:.4f}")
