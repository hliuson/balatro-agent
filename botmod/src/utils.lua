Utils = { }

function Utils.getCardData(card)
    local _card = { }

    _card.label = card.label
    _card.name = card.config.card.name
    _card.suit = card.config.card.suit
    _card.value = card.config.card.value
    _card.card_key = card.config.card_key

    return _card
end

function Utils.getDeckData()
    local _deck = { }

    return _deck
end

function Utils.getHandData()
    local _hand = { }

    if G and G.hand and G.hand.cards then
        for i = 1, #G.hand.cards do
            local _card = Utils.getCardData(G.hand.cards[i])
            _hand[i] = _card
        end
    end

    return _hand
end

function Utils.getJokersData()
    local _jokers = { }

    if G and G.jokers and G.jokers.cards then
        for i = 1, #G.jokers.cards do
            local _card = Utils.getCardData(G.jokers.cards[i])
            _jokers[i] = _card
        end
    end

    return _jokers
end

function Utils.getConsumablesData()
    local _consumables = { }

    if G and G.consumables and G.consumables.cards then
        for i = 1, #G.consumeables.cards do
            local _card = Utils.getCardData(G.consumeables.cards[i])
            _consumables[i] = _card
        end
    end

    return _consumables
end

function Utils.getBlindData()
    local _blinds = { }

    if G and G.GAME then
        _blinds.ondeck = G.GAME.blind_on_deck
    end

    return _blinds
end

function Utils.getAnteData()
    local _ante = { }
    _ante.blinds = Utils.getBlindData()

    return _ante
end

function Utils.getBackData()
    local _back = { }

    return _back
end

function Utils.getShopData()
    local _shop = { }
    if not G or not G.shop then return _shop end
    
    _shop.reroll_cost = G.GAME.current_round.reroll_cost
    _shop.cards = { }
    _shop.boosters = { }
    _shop.vouchers = { }

    for i = 1, #G.shop_jokers.cards do
        _shop.cards[i] = Utils.getCardData(G.shop_jokers.cards[i])
    end

    for i = 1, #G.shop_booster.cards do
        _shop.boosters[i] = Utils.getCardData(G.shop_booster.cards[i])
    end

    for i = 1, #G.shop_vouchers.cards do
        _shop.vouchers[i] = Utils.getCardData(G.shop_vouchers.cards[i])
    end

    return _shop
end

function Utils.getHandScoreData()
    local _handscores = { }

    return _handscores
end

function Utils.getTagsData()
    local _tags = { }

    return _tags
end

function Utils.getRoundData()
    local _current_round = { }

    if G and G.GAME and G.GAME.current_round then
        _current_round.discards_left = G.GAME.current_round.discards_left
        _current_round.hands_left = G.GAME.current_round.hands_left
    end

    return _current_round
end

function Utils.getGameData()
    local _game = { }

    if G and G.STATE then
        _game.state = G.STATE
        _game.num_hands_played = G.GAME.hands_played
        _game.num_skips = G.GAME.Skips
        _game.round = G.GAME.round
        _game.discount_percent = G.GAME.discount_percent
        _game.interest_cap = G.GAME.interest_cap
        _game.inflation = G.GAME.inflation
        _game.dollars = G.GAME.dollars
        _game.max_jokers = G.GAME.max_jokers
        _game.bankrupt_at = G.GAME.bankrupt_at
        _game.chips = _game.chips
    end

    return _game
end

function Utils.getGamestate()
    local _gamestate = {}

    _gamestate.state = G.STATE
    _gamestate.waiting_for = G.waitingFor
    
    -- Create a simplified version of G.GAME
    -- keys: 'tarot_rate', 'challenge_tab', 'pack_size', 'round_bonus', 'tag_tally', 'win_ante', 'inflation', 'spectral_rate', 'banned_keys', 'joker_rate', 'used_vouchers', 'modifiers', 'pseudorandom', 'selected_back_key', 'shop', 'STOP_USE', 'round_resets', 'unused_discards', 'disabled_suits', 'ecto_minus', 'selected_back', 'playing_card_rate', 'voucher_text', 'starting_params', 'round_scores', 'edition_rate', 'used_jokers', 'previous_round', 'blind', 'current_round', 'rental_rate', 'consumeable_usage', 'last_blind', 'planet_rate', 'stake', 'perscribed_bosses', 'interest_cap', 'perishable_rounds', 'skips', 'subhash', 'facing_blind', 'orbital_choices', 'interest_amount', 'chips_text', 'blind_on_deck', 'hands_played', 'disabled_ranks', 'round', 'hand_usage', 'consumeable_buffer', 'max_jokers', 'tags', 'base_reroll_cost', 'sort', 'discount_percent', 'pool_flags', 'bankrupt_at', 'joker_usage', 'cards_played', 'chips', 'dollars', 'current_boss_streak', 'won', 'hands', 'probabilities', 'starting_deck_size', 'bosses_used', 'joker_buffer', 'seeded'
    if G.GAME then
        _gamestate.game = {
            hands_played = G.GAME.hands_played,
            Skips = G.GAME.Skips,
            round = G.GAME.round,
            dollars = G.GAME.dollars,
            max_jokers = G.GAME.max_jokers,
            bankrupt_at = G.GAME.bankrupt_at,
            chips = G.GAME.chips,
        }
    end

    -- Use existing utility functions to get data
    _gamestate.hand = Utils.getHandData()
    _gamestate.jokers = Utils.getJokersData()
    _gamestate.consumables = Utils.getConsumablesData()
    _gamestate.shop = Utils.getShopData()
    _gamestate.round = Utils.getRoundData()
    return _gamestate
end

function Utils.getRoundData()
    local _round = { }

    if G and G.GAME and G.GAME.current_round then
        _round.discards_left = G.GAME.current_round.discards_left
        _round.hands_left = G.GAME.current_round.hands_left
        _round.blind_on_deck = G.GAME.blind_on_deck
        _round.reroll_cost = G.GAME.current_round.reroll_cost
    end

    return _round
end

function Utils.parseaction(data)
    -- Protocol is ACTION|arg1|arg2
    action = data:match("^([%a%u_]*)")
    params = data:match("|(.*)")

    if action then
        local _action = Bot.ACTIONS[action]

        if not _action then
            return nil
        end

        local _actiontable = { }
        _actiontable[1] = _action

        if params then
            local _i = 2
            for _arg in params:gmatch("[%w%s,]+") do
                local _splitstring = { }
                local _j = 1
                for _str in _arg:gmatch('([^,]+)') do
                    _splitstring[_j] = tonumber(_str) or _str
                    _j = _j + 1
                end
                _actiontable[_i] = _splitstring
                _i = _i + 1
            end
        end

        return _actiontable
    end
end

Utils.ERROR = {
    NOERROR = 1,
    NUMPARAMS = 2,
    MSGFORMAT = 3,
    INVALIDACTION = 4,
}

function Utils.validateAction(action)
    if action and #action > 1 and #action > Bot.ACTIONPARAMS[action[1]].num_args then
        return Utils.ERROR.NUMPARAMS
    elseif not action then
        return Utils.ERROR.MSGFORMAT
    else
        if not Bot.ACTIONPARAMS[action[1]].isvalid(action) then
            return Utils.ERROR.INVALIDACTION
        end
    end

    return Utils.ERROR.NOERROR
end

function Utils.isTableUnique(table)
    if table == nil then return true end

    local _seen = { }
    for i = 1, #table do
        if _seen[table[i]] then return false end
        _seen[table[i]] = table[i]
    end

    return true
end

function Utils.isTableInRange(table, min, max)
    if table == nil then return true end

    for i = 1, #table do
        if table[i] < min or table[i] > max then return false end
    end
    return true
end

return Utils