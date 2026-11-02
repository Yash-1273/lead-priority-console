"""Inject the trained model, data and experiment results into the console page.

Run after train.py and experiments.py:  python workflow/build_console.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / 'console_data.json').read_text(encoding='utf8'))
data['experiments'] = json.loads((HERE / 'experiments.json').read_text(encoding='utf8'))
html = (HERE / 'console_template.html').read_text(encoding='utf8')
(HERE.parent / 'Lead_Priority_Console.html').write_text(
    html.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf8')
print('Built Lead_Priority_Console.html')
