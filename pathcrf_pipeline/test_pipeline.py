"""Contract, geometry, temporal integrity and real-checkpoint integration checks."""
import unittest
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import fill_short_gaps, movement_features, make_window, window_schedule
from .goalkeepers import match_detection
from .rules import combine_predictions


def synthetic_tracking(n=50):
    team=np.repeat([0,1],11).astype(np.int8)
    xy=np.zeros((n,22,2),np.float32)
    xy[:,:,0]=np.r_[5,np.linspace(20,45,10),100,np.linspace(60,85,10)]/105
    xy[:,:,1]=np.tile(np.linspace(10,58,11),2)/68
    return xy,team


class InputTests(unittest.TestCase):
    def test_constant_velocity_and_acceleration(self):
        t=np.arange(100)/25
        xy=np.c_[20+3*t,30-4*t]
        f=movement_features(xy)
        np.testing.assert_allclose(f[10:-10,2:4],np.tile([3,-4],(80,1)),atol=1e-3)
        np.testing.assert_allclose(f[10:-10,4],5,atol=1e-3)
        np.testing.assert_allclose(f[10:-10,5],0,atol=.01)
        accelerated=movement_features(np.c_[20+t*t,np.full(100,30)])
        np.testing.assert_allclose(accelerated[15:-15,5],2,atol=.015)

    def test_no_extrapolation_or_long_gap_fabrication(self):
        xy=np.c_[np.arange(12),np.zeros(12)].astype(np.float32)
        xy[:2]=np.nan
        xy[4:6]=np.nan
        xy[8:11]=np.nan
        filled=fill_short_gaps(xy,max_gap=2)
        self.assertTrue(np.isnan(filled[:2]).all())
        np.testing.assert_allclose(filled[4:6,0],[4,5])
        self.assertTrue(np.isnan(filled[8:11]).all())

    def test_eight_features_team_blocks_and_boundary_nodes(self):
        xy,team=synthetic_tracking()
        prepared,info=make_window(xy,team)
        self.assertEqual(info['padded_nodes'],0)
        self.assertEqual(prepared['tensor'].shape,(50,26,8))
        np.testing.assert_array_equal(prepared['tensor'][0,:11,0],-1)
        np.testing.assert_array_equal(prepared['tensor'][0,11:22,0],1)
        self.assertEqual(prepared['tensor'][0,:,1].sum(),2)
        np.testing.assert_allclose(prepared['tensor'][0,22:,2:4],[[0,34],[105,34],[52.5,0],[52.5,68]])

    def test_identity_jump_is_excluded_not_differentiated(self):
        xy,team=synthetic_tracking()
        xy[25:,4,0]+=.3
        prepared,info=make_window(xy,team)
        self.assertIsNotNone(prepared)
        self.assertNotIn(4,prepared['node_slot'])
        self.assertEqual(info['padded_nodes'],1)

    def test_no_blind_truncation_of_excess_players(self):
        xy,team=synthetic_tracking()
        xy=np.concatenate([xy,np.tile([[[.48,.95]]],(len(xy),1,1))],axis=1)
        team=np.r_[team,0]
        prepared,info=make_window(xy,team)
        self.assertIsNone(prepared)
        self.assertEqual(info['reason'],'unresolved_excess_players')

    def test_window_coverage_does_not_cross_halftime(self):
        blocks=[(1,0,999),(2,2000,2499)]
        windows=window_schedule(blocks)
        covered=np.zeros(2500,bool)
        for half,a,b in windows:
            self.assertTrue((half==1 and b<=1000) or (half==2 and a>=2000))
            covered[a:b]=True
        self.assertTrue(covered[:1000].all())
        self.assertFalse(covered[1000:2000].any())
        self.assertTrue(covered[2000:].all())

    def test_numpy_segment_boundaries_have_serializable_audits(self):
        blocks=[(np.int64(1),np.int64(10),np.int64(49)),
                (np.int64(2),np.int64(1000),np.int64(1500))]
        windows=window_schedule(blocks)
        json.dumps([{'half':h,'start':a,'end_exclusive':b} for h,a,b in windows])
        self.assertTrue(all(type(value) is int for window in windows for value in window))

    def test_bottom_centre_inverse_homography_and_ambiguity(self):
        H=np.array([[1000,0,10],[0,500,20],[0,0,1]],float)
        box=[500,200,520,270]
        slot,distance,xy=match_detection(box,H,np.array([[.5,.5],[.8,.8]]))
        self.assertEqual(slot,0)
        self.assertAlmostEqual(distance,0)
        np.testing.assert_allclose(xy,[.5,.5])
        slot,_,_=match_detection(box,H,np.array([[.5,.5],[.5,.5]]))
        self.assertEqual(slot,-1)

    def test_model_coverage_and_fallback_remain_distinct(self):
        votes=np.array([[0,0,0],[0,1,0],[1,0,0]],np.float32)
        weights=np.array([0,1,1],np.float32)
        rules={'labels':np.array(['H','H','D']), 'dead':np.array([0,0,1],bool),
               'active':np.ones(3,bool)}
        pure,combined=combine_predictions(votes,weights,rules)
        np.testing.assert_array_equal(pure,['X','A','H'])
        np.testing.assert_array_equal(combined,['H','A','D'])


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .infer import load_model
        cls.model,_=load_model(4)

    def test_real_checkpoint_matches_official_inference(self):
        from inference import inference_episode
        from .infer import predict_window
        xy,team=synthetic_tracking()
        prepared,info=make_window(xy,team)
        columns={}
        names=[]
        for node in range(22):
            name=('home' if team[node]==0 else 'away')+f'_{node:03d}'
            names.append(name)
            for j,suffix in enumerate(['x','y','vx','vy','speed','accel']):
                columns[f'{name}_{suffix}']=prepared['tensor'][:,node,2+j]
        tracking=pd.DataFrame(columns)
        _,_,official,order,_=inference_episode(self.model,tracking,
                         left_gk=names[0],right_gk=names[11],evaluate=False,device='cpu')
        labels,path=predict_window(self.model,prepared)
        outside=['out_left','out_right','out_bottom','out_top']
        all_names=names+outside
        expected_src=[all_names[i] for i in path['source_node']]
        expected_dst=[all_names[i] for i in path['destination_node']]
        sample=path['sample_frame']
        self.assertEqual(official.loc[sample,'edge_src'].tolist(),expected_src)
        self.assertEqual(official.loc[sample,'edge_dst'].tolist(),expected_dst)
        labels2,path2=predict_window(self.model,prepared)
        np.testing.assert_array_equal(labels,labels2)
        np.testing.assert_array_equal(path['source_node'],path2['source_node'])
        self.assertTrue(all(not p.requires_grad or p.grad is None for p in self.model.parameters()))

    def test_prediction_guard_blocks_answer_keys(self):
        from .run import install_ground_truth_guard
        install_ground_truth_guard()
        path=Path(__file__).resolve().parents[1]/'evaluation'/'gt'/'BAR-ATH_possession_gt_public.csv'
        with self.assertRaises(PermissionError):
            path.read_text()

    def test_all_variants_match_the_official_decoder(self):
        from inference import inference_episode
        from .infer import load_model, predict_window
        xy,team=synthetic_tracking()
        prepared,_=make_window(xy,team)
        names=[('home' if team[node]==0 else 'away')+f'_{node:03d}' for node in range(22)]
        columns={f'{names[node]}_{suffix}':prepared['tensor'][:,node,2+j]
                 for node in range(22)
                 for j,suffix in enumerate(['x','y','vx','vy','speed','accel'])}
        all_names=names+['out_left','out_right','out_bottom','out_top']
        for trial in [100,110,120,130,140]:
            with self.subTest(trial=trial):
                model,_=load_model(4,trial)
                use_crf=model.crf is not None
                _,_,official,_,_=inference_episode(model,pd.DataFrame(columns),
                    left_gk=names[0],right_gk=names[11],evaluate=False,device='cpu',use_crf=use_crf)
                _,path=predict_window(model,prepared)
                sample=path['sample_frame']
                if use_crf:
                    actual_src=official.loc[sample,'edge_src'].tolist()
                    actual_dst=official.loc[sample,'edge_dst'].tolist()
                else:
                    pairs=[name.split('-') for name in official.loc[sample].idxmax(axis=1)]
                    actual_src=[p[0] for p in pairs]
                    actual_dst=[p[1] for p in pairs]
                self.assertEqual(actual_src,[all_names[i] for i in path['source_node']])
                self.assertEqual(actual_dst,[all_names[i] for i in path['destination_node']])
                partial_xy=xy.copy()
                partial_xy[:,3]=np.nan
                partial,_=make_window(partial_xy,team)
                _,partial_path=predict_window(model,partial)
                self.assertTrue(partial['active'][partial_path['source_node']].all())
                self.assertTrue(partial['active'][partial_path['destination_node']].all())

    def test_cpu_denormal_policy_preserves_logits_and_paths(self):
        import torch
        from .infer import load_model, predict_window
        xy,team=synthetic_tracking(250)
        time=np.arange(250,dtype=np.float32)/25
        xy[:,:,0] += (np.sin(time*.5)*.01)[:,None]
        prepared,_=make_window(xy,team)
        index=np.r_[np.arange(0,250,5),249]
        x=torch.from_numpy(prepared['tensor'][index]).unsqueeze(0)
        for trial in [100,110,120,130,140]:
            with self.subTest(trial=trial), torch.inference_mode():
                model,_=load_model(4,trial)
                torch.set_flush_denormal(False)
                _,reference,_=model(x)
                reference_labels,reference_path=predict_window(model,prepared)
                torch.set_flush_denormal(True)
                _,optimized,_=model(x)
                labels,path=predict_window(model,prepared)
                torch.testing.assert_close(reference,optimized,rtol=1e-6,atol=1e-6)
                np.testing.assert_array_equal(reference_labels,labels)
                np.testing.assert_array_equal(reference_path['source_node'],path['source_node'])
                np.testing.assert_array_equal(reference_path['destination_node'],path['destination_node'])


if __name__=='__main__':
    unittest.main()

