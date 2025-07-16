interface GameState {
  action_no: number;
  hand: Card[];
  waitingForAction: boolean;
  round: RoundInfo;
  deck: Deck;
  jokers: any[];
  hand_scores: HandScores;
  tags: any[];
  shop: Shop;
  state: number;
  consumables: any[];
  back: any[];
  game: GameInfo;
  ante: AnteInfo;
}

interface Card {
  cost: number;
  ability_name: string;
  sell_cost: number;
  label: string;
  chips: number;
  suit: 'Hearts' | 'Clubs' | 'Spades' | 'Diamonds';
  center_key: string;
  mult: number;
  name: string;
  card_key: string;  // Format: "S_A" (Suit_Value)
  set: string;
  value: 'Ace' | 'King' | 'Queen' | 'Jack' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' | '10';
  x_mult: number;
  debuff: boolean;
  perma_bonus: number;
}

interface RoundInfo {
  discards_left: number;
  reroll_cost: number;
  hands_played_this_round: number;
  free_rerolls: number;
  blind_on_deck: string;
  discards_used_this_round: number;
  hands_left: number;
}

interface Deck {
  discards: Card[];
  cards: Card[];
}

interface HandScores {
  [handType: string]: {
    level: number;
    mult: number;
    chips: number;
  };
}

interface Shop {
  planets: any[];
  playing_cards: any[];
  tarots: any[];
  jokers: any[];
  boosters: any[];
  vouchers: any[];
}

interface GameInfo {
  hands_played: number;
  max_jokers: number;
  bankrupt_at: number;
  inflation: number;
  current_streak: number;
  round: number;
  dollars: number;
  chips: number;
  stake: number;
  discount_percent: number;
  skips: number;
  interest_amount: number;
  state: number;
  win_streak: number;
  ante: number;
  interest_cap: number;
}

interface AnteInfo {
  blind_ante: number;
  win_ante: number;
  ante: number;
  blinds: {
    ondeck_key: string;
    small: Blind;
    current: Blind & { disabled: boolean; boss: boolean };
    big: Blind;
    boss: BossBlind;
  };
  blind_states: {
    [key: string]: 'Current' | 'Upcoming';
  };
}

interface Blind {
  mult: number;
  name: string;
  dollars: number;
  key: string;
  debuff: any[];
  chips: number;
}

interface BossBlind extends Blind {
  boss: {
    max: number;
    min: number;
  };
  debuff: {
    h_size_ge?: number;
  };
}