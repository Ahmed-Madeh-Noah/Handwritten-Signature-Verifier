from dataclasses import dataclass


@dataclass
class CONFIG:
    dataset = '../signatures'
    original_signatures_class = 'full_org'
    forged_signatures_class = 'full_forg'
    size = (350, 543)
    mean = 0.015957806259393692
    std = 0.08474896848201752
