"""Reproduce the lab-attention translation heatmap before and after partial RoPE.

Data, labels, and SVG styling are carried over from lab-attention/attention.py.
Only the first six of seven coordinates rotate; scores retain sqrt(7) scaling.
"""

import math
from pathlib import Path

source_tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
target_prefixes = ["BOS", "BOS Кіт", "BOS Кіт сидів",
                   "BOS Кіт сидів на", "BOS Кіт сидів на килимку"]
next_tokens = ["Кіт", "сидів", "на", "килимку", "."]
source_keys = [[int(i == j) for j in range(7)] for i in range(7)]
target_queries = [
    [0, 8, 2, 0, 0, 0, 0],
    [0, 2, 8, 2, 0, 0, 0],
    [0, 0, 2, 8, 1, 2, 0],
    [0, 0, 0, 2, 3, 8, 0],
    [0, 0, 1, 0, 0, 2, 8],
]


def source_attention_svg(weights, title="Cross-attention: The cat sat on the mat."):
    """Heatmap без додаткових бібліотек для побудови графіків."""
    from html import escape

    left, top, size = 280, 100, 70
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="530" '
             'viewBox="0 0 960 530" role="img" '
             f'aria-label="{escape(title)}">',
             '<rect width="960" height="530" fill="white"/>',
             '<g font-family="sans-serif" fill="#233246">',
             f'<text x="30" y="35" font-size="22">{escape(title)}</text>',
             '<text x="30" y="78" font-size="15">Поточний префікс / запит</text>',
             '<text x="800" y="78" font-size="15">Наступний токен</text>']
    for j, token in enumerate(source_tokens):
        parts.append(f'<text x="{left + j*size + size/2}" y="78" '
                     f'text-anchor="middle" font-size="17">{escape(token)}</text>')
    for i, row in enumerate(weights):
        y = top + i*size
        parts.append(f'<text x="265" y="{y+40}" text-anchor="end" '
                     f'font-size="15">{escape(target_prefixes[i])}</text>')
        parts.append(f'<text x="815" y="{y+40}" font-size="17">'
                     f'{escape(next_tokens[i])}</text>')
        for j, weight in enumerate(row):
            shade = [round(a + (b-a)*weight) for a, b in
                     zip((239, 246, 255), (25, 75, 155))]
            color = '#' + ''.join(f'{v:02x}' for v in shade)
            x = left + j*size
            ink = 'white' if weight > 0.55 else '#233246'
            parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" '
                         f'fill="{color}" stroke="white"/>')
            parts.append(f'<text x="{x+size/2}" y="{y+40}" text-anchor="middle" '
                         f'font-size="16" fill="{ink}">{weight:.3f}</text>')
    for j in range(100):
        weight = j/99
        shade = [round(a + (b-a)*weight) for a, b in
                 zip((239, 246, 255), (25, 75, 155))]
        color = '#' + ''.join(f'{v:02x}' for v in shade)
        parts.append(f'<rect x="{left+j*3}" y="480" width="3" height="12" fill="{color}"/>')
    parts.extend(['<text x="280" y="515" font-size="14">0</text>',
                  '<text x="570" y="515" font-size="14">1</text>',
                  '<text x="610" y="491" font-size="14">Вага attention</text>',
                  '</g></svg>'])
    return ''.join(parts)

def translation_weights(rotate=None):
    """Rows are decoder steps (BOS at 0); columns are source positions (0..6)."""
    def positioned(vector, position):
        if rotate is None:
            return vector[:]
        return rotate(vector[:6], position) + vector[6:]

    keys = [positioned(key, p) for p, key in enumerate(source_keys)]
    rows = []
    for p, query in enumerate(target_queries):
        query = positioned(query, p)
        scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(7)
                  for key in keys]
        exps = [math.exp(score - max(scores)) for score in scores]
        rows.append([value / sum(exps) for value in exps])
    return rows


def figure_rotation(vector, position):
    """Independent complex-number rotation for generating static figures."""
    from cmath import exp

    result = []
    for i in range(len(vector) // 2):
        pair = complex(*vector[2 * i:2 * i + 2])
        rotated = pair * exp(1j * position / 10000 ** (2 * i / len(vector)))
        result.extend((rotated.real, rotated.imag))
    return result


def main():
    output = Path(__file__).resolve().parents[1] / "figures"
    output.mkdir(exist_ok=True)
    for name, title, rotation in [
        ("source_attention.svg", "Без RoPE: The cat sat on the mat.", None),
        ("source_attention_rope.svg", "З RoPE: The cat sat on the mat.", figure_rotation),
    ]:
        (output / name).write_text(
            source_attention_svg(translation_weights(rotation), title), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
