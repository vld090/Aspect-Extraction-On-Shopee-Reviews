import csv
import json
from collections import defaultdict

INPUT_CSV = '001-050 _ Thesis Annotation Sheet - Trial.csv'  # Using trial CSV
CONLL_OUT = 'output.conll'
JSONL_OUT = 'output.jsonl'

# Read and group by review number
reviews = defaultdict(list)

def split_final_tag(final_tag):
    if not final_tag or final_tag.strip() == '':
        return '', ''
    parts = final_tag.split('-')
    if len(parts) == 2:
        bio = parts[0].strip()
        aspect = parts[1].strip()
        return bio, aspect
    # If not in expected format, return as is
    return final_tag.strip(), ''

with open(INPUT_CSV, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        review_id = row['Review #'] or None
        token = row['Token']
        final_expl = row['Final Tag (For Explicit Aspects)']
        final_impl = row['Final Tag (For Implicit Aspects)']
        expl_bio, expl_aspect = split_final_tag(final_expl)
        _, impl_aspect = split_final_tag(final_impl)
        if review_id and token:
            reviews[review_id].append((token, expl_bio, expl_aspect, impl_aspect))

# Write CoNLL format (explicit only, for reference)
with open(CONLL_OUT, 'w', encoding='utf-8') as f:
    for review in reviews.values():
        for token, expl_bio, expl_aspect, _ in review:
            f.write(f"{token} {expl_bio} {expl_aspect}\n")
        f.write("\n")  # Blank line between reviews

# Write JSONL format with separated BIO and aspect tags (no impl_bio)
with open(JSONL_OUT, 'w', encoding='utf-8') as f:
    for review in reviews.values():
        tokens = [token for token, _, _, _ in review]
        expl_bio = [expl_bio for _, expl_bio, _, _ in review]
        expl_aspect = [expl_aspect for _, _, expl_aspect, _ in review]
        impl_aspect = [impl_aspect for _, _, _, impl_aspect in review]
        json.dump({
            'tokens': tokens,
            'expl_bio': expl_bio,
            'expl_aspect': expl_aspect,
            'impl_aspect': impl_aspect
        }, f, ensure_ascii=False)
        f.write('\n')

print(f"Wrote {len(reviews)} reviews to {CONLL_OUT} and {JSONL_OUT} (with explicit BIO/aspect and implicit aspect only)") 