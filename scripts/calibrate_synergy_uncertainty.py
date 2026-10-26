"""Estimate descriptive uncertainty proxies from repeated glove poses."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.glove.captured_samples import load_captured_samples
from src.hand_model.robot_hand import RobotHandModel
from src.retargeting.latent_synergy import LatentSynergyMapper
from src.retargeting.adaptive_synergy import TaskAdaptiveSynergyMapper
from src.retargeting.uncertainty_model import fit_uncertainty_proxy


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('configs/latent_chm6_adaptive.json'))
    p.add_argument('--out', type=Path, default=Path('configs/chm6_uncertainty_proxy.json'))
    a = p.parse_args()
    latent = LatentSynergyMapper(RobotHandModel(), a.model)
    mapper = TaskAdaptiveSynergyMapper(latent, {})
    config = fit_uncertainty_proxy(load_captured_samples(a.samples), mapper)
    config['source_capture'] = str(a.samples)
    config['encoder_model'] = str(a.model)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(config, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'Wrote descriptive proxy {a.out} ({config["source_rows"]} rows); not a coverage certificate.')

if __name__ == '__main__':
    main()
