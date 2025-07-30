"""Simple vocabularies for Balatro cards."""

import json

# Load all vocabularies at module level
with open('jokers_vocab.json') as f:
    JOKERS = json.load(f)

with open('tarots_vocab_fixed.json') as f:
    TAROTS = json.load(f)
    
with open('planets_vocab_fixed.json') as f:
    PLANETS = json.load(f)
    
with open('spectrals_vocab_fixed.json') as f:
    SPECTRALS = json.load(f)

# Create name->index mappings
JOKER_VOCAB = {name: i+1 for i, name in enumerate(JOKERS)}
TAROT_VOCAB = {name: i+1 for i, name in enumerate(TAROTS)}
PLANET_VOCAB = {name: i+1 for i, name in enumerate(PLANETS)}
SPECTRAL_VOCAB = {name: i+1 for i, name in enumerate(SPECTRALS)}

# Add unknown tokens
for vocab in [JOKER_VOCAB, TAROT_VOCAB, PLANET_VOCAB, SPECTRAL_VOCAB]:
    vocab['<UNK>'] = 0

# Embedding sizes
VOCAB_SIZES = {
    'jokers': len(JOKER_VOCAB),
    'tarots': len(TAROT_VOCAB), 
    'planets': len(PLANET_VOCAB),
    'spectrals': len(SPECTRAL_VOCAB),
    'ranks': 13,  # A-K
    'suits': 4,   # SHDC
}