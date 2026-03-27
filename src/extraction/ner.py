from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class _NatashaBundle:
    segmenter: object
    morph_vocab: object
    emb: object
    morph_tagger: object
    syntax_parser: object
    ner_tagger: object


def _load_natasha() -> _NatashaBundle | None:
    try:
        from natasha import (
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsMorphTagger,
            NewsNERTagger,
            NewsSyntaxParser,
            Segmenter,
        )
    except Exception:
        return None

    bundle = _NatashaBundle(
        segmenter=Segmenter(),
        morph_vocab=MorphVocab(),
        emb=NewsEmbedding(),
        morph_tagger=NewsMorphTagger(NewsEmbedding()),
        syntax_parser=NewsSyntaxParser(NewsEmbedding()),
        ner_tagger=NewsNERTagger(NewsEmbedding()),
    )
    bundle._doc_class = Doc  # type: ignore[attr-defined]
    return bundle


def normalize_entity(entity: str) -> str:
    entity = re.sub(r"\s+", " ", entity).strip()
    return entity


def extract_entities(text: str) -> dict[str, list[str]]:
    entities: dict[str, list[str]] = {"Person": [], "Organization": [], "Date": []}

    bundle = _load_natasha()
    if bundle is not None:
        Doc = bundle._doc_class  # type: ignore[attr-defined]
        doc = Doc(text)
        doc.segment(bundle.segmenter)
        doc.tag_morph(bundle.morph_tagger)
        doc.parse_syntax(bundle.syntax_parser)
        doc.tag_ner(bundle.ner_tagger)

        for span in doc.spans:
            value = normalize_entity(span.text)
            if not value:
                continue
            if span.type == "PER":
                entities["Person"].append(value)
            elif span.type == "ORG":
                entities["Organization"].append(value)
            elif span.type == "DATE":
                entities["Date"].append(value)

    date_pattern = re.compile(r"\b(20\d{2}[-/.]\d{2}[-/.]\d{2}|\d{2}[-/.]\d{2}[-/.]20\d{2})\b")
    for m in date_pattern.findall(text):
        entities["Date"].append(normalize_entity(m))

    title_name_pattern = re.compile(r"\b([A-ZА-Я][a-zа-я]+\s+[A-ZА-Я][a-zа-я]+)\b")
    for m in title_name_pattern.findall(text):
        if m not in entities["Person"]:
            entities["Person"].append(normalize_entity(m))

    for key in entities:
        entities[key] = sorted(set(entities[key]))
    return entities
