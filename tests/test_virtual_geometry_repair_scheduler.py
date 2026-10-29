"""CPU controller checks using predicted depth, never an A-model selection mask."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('repair_scheduler_cpu', ROOT / 'scene/virtual_geometry_priors.py')
virtual = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(virtual)


class RepairSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        self.assertFalse(torch.cuda.is_initialized())

    def fixture(self, shared=False, two_stages=False):
        for name, value in [('depth', .8), ('mask', 1.), ('lower', .784), ('upper', .816)]:
            np.save(self.root / (name + '.npy'), np.full((20, 20), value, np.float32))
        definitions = [('weak', 'p0', 'weak'), ('stable', 'p0', 'repair'), ('few_bad', 'p1', 'repair'), ('repair_needed', 'p2', 'repair')]
        if shared:
            definitions = [('weak', 'p0', 'weak'), ('repair_needed', 'p0', 'repair'), ('other_weak', 'p9', 'weak')]
        views = []
        for name, physical, mode in definitions:
            view = {'image_name': name, 'target_id': physical + '_' + mode, 'physical_patch_id': physical,
                    'spatial_group': physical, 'source_mode': mode, 'priority': 1., 'loss_weight': .5,
                    'target_priority': 1., 'width': 20, 'height': 20,
                    'K': [[20., 0, 10.], [0, 20., 10.], [0, 0, 1]], 'c2w': np.eye(4).tolist(),
                    'depth_path': str(self.root / 'depth.npy'), 'mask_path': str(self.root / 'mask.npy')}
            if mode == 'repair':
                view.update(interval_lower_path=str(self.root / 'lower.npy'), interval_upper_path=str(self.root / 'upper.npy'))
            views.append(view)
        manifest = {'schema_version': 1, 'depth_convention': 'camera_z', 'units': 'scene',
                    'pixel_center_convention': 'colmap_half_integer', 'views': views}
        mp = self.root / 'manifest.json'; mp.write_text(json.dumps(manifest))
        config = {'schema_version': 1, 'manifest': str(mp), 'weight': .3, 'from_iter': 5, 'ramp_iters': 0,
                  'every_n_real_updates': 1, 'loss': {'deadband_relative': .002, 'robust_saturation': .05},
                  'interval_repair': {'enabled': True},
                  'scheduler': {'real_camera_count': 9, 'activation_iterations': [5, 8] if two_stages else [5],
                                'batch_sizes': [4, 4] if two_stages else [4], 'local_ramp_iters': 0,
                                'target_budget': 'bounded_sqrt_priority', 'probe_every': 100,
                                'repair_pre_activation_probe': True, 'repair_probe_limit': 2000,
                                'repair_min_excess_fraction': .01, 'repair_min_mean_relative_excess': 1e-5}}
        cp = self.root / 'config.json'; cp.write_text(json.dumps(config)); return cp

    def feedback(self, priors, view, wake_stable=False):
        pred = view['depth_cpu'].clone()
        if view['image_name'] == 'few_bad':
            pred.reshape(-1)[:3] = 0.  # .75%, below the 1% camera activation gate
        elif view['image_name'] == 'repair_needed' or (wake_stable and view['image_name'] == 'stable'):
            pred.reshape(-1)[:4] *= 1.04  # exactly 1% outside the fixed 2% interval
        else:
            pred.reshape(-1)[::2] *= 1.01
            pred.reshape(-1)[1::2] *= .99  # in-band roughness cannot manufacture repair need
        return priors.probe_feedback(view, pred, view['depth_cpu'], view['mask_cpu'])

    def test_probe_precedes_activation_and_one_percent_gate_keeps_stable_views_dormant(self):
        priors = virtual.VirtualGeometryPriors(self.fixture())
        priors.maybe_update_schedule(5, lambda view: self.feedback(priors, view))
        names = {priors.views[index]['image_name'] for index in priors.active_indices}
        self.assertEqual(names, {'weak', 'repair_needed'})
        self.assertEqual(priors.record['scheduler']['probe_render_samples'], 3)
        summary = priors.record['scheduler']['repair_pre_activation_events'][0]
        self.assertEqual(summary['repair_dormant_after_probe'], 2)
        self.assertEqual(summary['activated_repair_views'], 1)
        self.assertEqual(priors.probe_latest['stable']['need_rank'], 0.)
        self.assertGreater(priors.probe_latest['stable']['relative_center_abs_error'], .009)
        self.assertEqual(priors.probe_latest['stable']['relative_interval_excess'], 0.)

    def test_modes_of_same_physical_patch_share_one_ticket_budget(self):
        priors = virtual.VirtualGeometryPriors(self.fixture(shared=True))
        priors.maybe_update_schedule(5, lambda view: self.feedback(priors, view))
        self.assertEqual(set(priors.groups), {'p0', 'p9'})
        self.assertEqual(priors.record['scheduler']['target_budget_tickets'], {'p0': 2, 'p9': 2})
        counts = {'p0': 0, 'p9': 0}
        for _ in range(200):
            view = priors.draw(device='cpu')[0]
            counts[view['target_id']] += 1
        self.assertEqual(counts, {'p0': 100, 'p9': 100})

    def test_integrated_repair_in_band_has_no_auxiliary_gradient(self):
        priors = virtual.VirtualGeometryPriors(self.fixture())
        view = next(view for view in priors.views if view['image_name'] == 'stable')
        pred = (view['depth_cpu'] * 1.01).requires_grad_()
        loss = priors.depth_loss_for(view, pred, view['depth_cpu'], view['mask_cpu'])
        loss.backward()
        self.assertEqual(float(loss.detach()), 0.)
        self.assertTrue((pred.grad == 0).all())

    def test_repair_need_can_outweigh_greedy_order_without_changing_source_confidence(self):
        cp = self.fixture()
        config = json.loads(cp.read_text())
        mp = Path(config['manifest'])
        manifest = json.loads(mp.read_text())
        # Same physical/spatial load and offline area-quality; the planner's
        # first camera has less current need than its later camera.
        manifest['views'] = [v for v in manifest['views'] if v['image_name'] in ('few_bad', 'repair_needed')]
        for view in manifest['views']:
            view['physical_patch_id'] = view['spatial_group'] = 'same_patch'
            view['offline_quality_priority'] = 3.
            view['priority'] = 1. if view['image_name'] == 'few_bad' else .1
        mp.write_text(json.dumps(manifest))
        config['scheduler']['batch_sizes'] = [1]
        cp.write_text(json.dumps(config))
        priors = virtual.VirtualGeometryPriors(cp)
        def feedback(view):
            pred = view['depth_cpu'].clone()
            pred *= 1.03 if view['image_name'] == 'few_bad' else 1.10
            return priors.probe_feedback(view, pred, view['depth_cpu'], view['mask_cpu'])
        priors.maybe_update_schedule(5, feedback)
        self.assertEqual([priors.views[i]['image_name'] for i in priors.active_indices], ['repair_needed'])
        self.assertTrue(all(view['loss_weight'] == .5 for view in priors.views))
        self.assertTrue(all(view['offline_quality_priority'] == 3. for view in priors.views))

    def test_interval_weak_and_repair_use_comparable_offline_area_quality_units(self):
        cp = self.fixture(shared=True)
        config = json.loads(cp.read_text())
        mp = Path(config['manifest'])
        manifest = json.loads(mp.read_text())
        manifest['views'] = [v for v in manifest['views'] if v['image_name'] in ('weak', 'repair_needed')]
        for view in manifest['views']:
            view['priority'] = 1. if view['source_mode'] == 'weak' else .1
            view['offline_quality_priority'] = .01 if view['source_mode'] == 'weak' else .3
        mp.write_text(json.dumps(manifest))
        config['scheduler']['batch_sizes'] = [1]
        cp.write_text(json.dumps(config))
        priors = virtual.VirtualGeometryPriors(cp)
        def feedback(view):
            pred = view['depth_cpu'] * 1.10
            return priors.probe_feedback(view, pred, view['depth_cpu'], view['mask_cpu'])
        priors.maybe_update_schedule(5, feedback)
        self.assertEqual([priors.views[i]['image_name'] for i in priors.active_indices], ['repair_needed'])
        self.assertTrue(all(view['loss_weight'] == .5 for view in priors.views))

    def test_area_quality_change_is_opt_in_and_legacy_weak_order_is_preserved(self):
        cp = self.fixture(shared=True)
        config = json.loads(cp.read_text())
        mp = Path(config['manifest'])
        manifest = json.loads(mp.read_text())
        manifest['views'] = [v for v in manifest['views'] if v['source_mode'] == 'weak']
        for view in manifest['views']:
            first = view['image_name'] == 'weak'
            view.update(priority=1. if first else .1, offline_quality_priority=.01 if first else .3,
                        physical_patch_id='same_patch', spatial_group='same_patch')
        mp.write_text(json.dumps(manifest))
        config['scheduler']['batch_sizes'] = [1]
        for interval_enabled, expected in [(True, 'other_weak'), (False, 'weak')]:
            config['interval_repair']['enabled'] = interval_enabled
            config['scheduler']['repair_pre_activation_probe'] = interval_enabled
            cp.write_text(json.dumps(config))
            priors = virtual.VirtualGeometryPriors(cp)
            priors.maybe_update_schedule(5)
            self.assertEqual([priors.views[i]['image_name'] for i in priors.active_indices], [expected])

    def test_dormant_repair_can_wake_from_new_gs_feedback_after_checkpoint_resume(self):
        cp = self.fixture(two_stages=True)
        priors = virtual.VirtualGeometryPriors(cp)
        priors.maybe_update_schedule(5, lambda view: self.feedback(priors, view))
        state = json.loads(json.dumps(priors.checkpoint_state()))
        restored = virtual.VirtualGeometryPriors(cp, checkpoint_state=state)
        for current in (priors, restored):
            current.maybe_update_schedule(8, lambda view: self.feedback(current, view, wake_stable=True))
        self.assertEqual({priors.views[index]['image_name'] for index in priors.active_indices}, {'weak', 'repair_needed', 'stable'})
        self.assertEqual(json.loads(json.dumps(priors.checkpoint_state())), json.loads(json.dumps(restored.checkpoint_state())))


if __name__ == '__main__':
    unittest.main()
