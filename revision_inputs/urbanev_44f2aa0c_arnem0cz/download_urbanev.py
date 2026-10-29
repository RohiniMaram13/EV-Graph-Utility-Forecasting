"""Download a versioned UrbanEV data snapshot and inspect it; no training.
Run in your Ubuntu project folder: python -u download_urbanev.py
Source: https://github.com/IntelligentSystemsLab/UrbanEV (CC0)
This preserves upstream data unchanged. The published energy volume is an
estimate based on charging-pile power, not a direct metered-energy measurement.
Upstream preprocessing includes imputation; this script does not undo it.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import Request, urlopen


def fetch(url):
    request = Request(url, headers={'User-Agent': 'UrbanEV-research-data-audit'})
    with urlopen(request, timeout=60) as response:
        return response.read()


def main():
    import numpy as np
    import pandas as pd
    root = Path.cwd()
    if not (root / 'real_validation').is_dir():
        raise RuntimeError('Run this from your existing forecasting project folder')
    metadata = json.loads(fetch('https://api.github.com/repos/IntelligentSystemsLab/UrbanEV/commits/main'))
    commit = metadata['sha']
    assert re.fullmatch(r'[0-9a-f]{40}', commit), commit
    parent = root / 'revision_inputs'
    parent.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f'urbanev_{commit[:8]}_', dir=parent))
    paths = ['readme.md', 'LICENSE', 'data/volume.csv', 'data/duration.csv',
             'data/inf.csv', 'data/adj.csv', 'data/distance.csv']
    record = {'repository': 'https://github.com/IntelligentSystemsLab/UrbanEV',
              'commit': commit, 'files': {}}
    report = [f'UrbanEV source commit: {commit}', f'Local snapshot: {output}',
              'Read-only audit of published data; no model training or graph selection.',
              'Caution: volume is estimated energy, not directly metered energy.',
              'Upstream preprocessing includes imputation; document this limitation.']
    shutil.copy2(Path(__file__).resolve(), output / 'download_urbanev.py')
    for relative in paths:
        url = f'https://raw.githubusercontent.com/IntelligentSystemsLab/UrbanEV/{commit}/{relative}'
        print(f'Downloading {relative} ...', flush=True)
        payload = fetch(url)
        if payload.startswith(b'version https://git-lfs.github.com/spec'):
            raise RuntimeError(f'{relative} is an LFS pointer, not data; stop and report this')
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        record['files'][relative] = {'url': url, 'bytes': len(payload),
                                    'sha256': hashlib.sha256(payload).hexdigest()}
        (output / 'download_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
        if target.suffix == '.csv':
            frame = pd.read_csv(target)
            report += [f'\n{relative}: rows={len(frame)}, columns={len(frame.columns)}',
                       f'First column names: {list(frame.columns[:8])}',
                       'First two rows, first six columns:', frame.iloc[:2, :6].to_string(index=False),
                       f'Missing cells: {int(frame.isna().sum().sum())}']
            if relative in ('data/volume.csv', 'data/duration.csv'):
                times = pd.to_datetime(frame.iloc[:, 0], errors='coerce')
                report += [f'First-column timestamp parse failures: {int(times.isna().sum())}']
                if times.notna().all():
                    report += [f'Time range: {times.iloc[0]} to {times.iloc[-1]}',
                               f'Duplicate timestamps: {int(times.duplicated().sum())}',
                               f'Time-step counts: {times.diff().dropna().value_counts().to_dict()}']
                values = frame.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
                finite = np.isfinite(values)
                report += [f'Nonfinite or nonnumeric value cells: {int((~finite).sum())}',
                           f'Negative value cells: {int((values < 0).sum())}',
                           f'Zero value cells: {int((values == 0).sum())}']
    report += ['\nDOWNLOAD COMPLETE. No forecasting experiment has been run.',
               'Next: verify node-ID alignment and geographic matrix meaning; freeze target,',
               'chronological splits and graph rule before any test-model evaluation.',
               'Keep the entire snapshot folder and its manifest for reproducibility.']
    text = '\n'.join(report) + '\n'
    (output / 'inspection_report.txt').write_text(text)
    print('\n' + text, flush=True)


if __name__ == '__main__':
    main()
