#!/usr/bin/env python3
"""Cache English Wikidata labels used by human-readable evaluation exports.

The default mode resolves identifiers referenced in QA CSV files plus every
relation in relation2id.txt. Use --all-entities to resolve the full KG entity
vocabulary as well, which is slower but labels arbitrary predicted endpoints.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ENTITY_PATTERN = re.compile(r'\bQ\d+\b')
RELATION_PATTERN = re.compile(r'\bP\d+\b')
API_URL = 'https://www.wikidata.org/w/api.php'


def index_identifiers(path: Path) -> set[str]:
    return {
        line.split('\t', 1)[0].removesuffix('_inv')
        for line in path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    }


def qa_identifiers(data_dir: Path) -> tuple[set[str], set[str]]:
    entities: set[str] = set()
    relations: set[str] = set()
    for csv_path in data_dir.glob('*.csv'):
        text = csv_path.read_text(encoding='utf-8')
        entities.update(ENTITY_PATTERN.findall(text))
        relations.update(RELATION_PATTERN.findall(text))
    return entities, relations


def chunks(values: Iterable[str], size: int = 50) -> Iterable[list[str]]:
    batch: list[str] = []
    for value in sorted(values):
        batch.append(value)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def fetch_labels(identifiers: set[str], pause_seconds: float) -> dict[str, str]:
    labels: dict[str, str] = {}
    for batch in chunks(identifiers):
        query = urlencode({
            'action': 'wbgetentities',
            'ids': '|'.join(batch),
            'props': 'labels',
            'languages': 'en',
            'format': 'json',
        })
        request = Request(
            '{}?{}'.format(API_URL, query),
            headers={'User-Agent': 'salesforce-kgqa-label-cache/1.0'},
        )
        with urlopen(request, timeout=60) as response:  # nosec B310: fixed HTTPS endpoint
            payload = json.load(response)
        for identifier, entity in payload.get('entities', {}).items():
            label = entity.get('labels', {}).get('en', {}).get('value')
            if label:
                labels[identifier] = str(label)
        if pause_seconds:
            time.sleep(pause_seconds)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', required=True, type=Path)
    parser.add_argument('--all-entities', action='store_true')
    parser.add_argument('--pause-seconds', type=float, default=0.05)
    args = parser.parse_args()

    output_path = args.data_dir / 'wikidata_labels.json'
    cached = {'entities': {}, 'relations': {}}
    if output_path.is_file():
        cached.update(json.loads(output_path.read_text(encoding='utf-8')))

    qa_entities, qa_relations = qa_identifiers(args.data_dir)
    entities = (index_identifiers(args.data_dir / 'entity2id.txt')
                if args.all_entities else qa_entities)
    relations = index_identifiers(args.data_dir / 'relation2id.txt') | qa_relations
    cached['entities'].update(fetch_labels(entities - set(cached['entities']), args.pause_seconds))
    cached['relations'].update(fetch_labels(relations - set(cached['relations']), args.pause_seconds))
    output_path.write_text(json.dumps(cached, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('Wrote {} entity and {} relation labels to {}'.format(
        len(cached['entities']), len(cached['relations']), output_path
    ))


if __name__ == '__main__':
    main()
