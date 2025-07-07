Utils = {}

-- Safely gets a value from a nested table
function safe_get(t, keys)
    local current = t
    for i, key in ipairs(keys) do
        if type(current) ~= 'table' or current[key] == nil then
            return nil
        end
        current = current[key]
    end
    return current
end

function Utils.getCardData(card)
    if not card then return nil end
    local _card = {}

    _card.label = card.label
    _card.name = safe_get(card, { 'ability', 'name' }) or safe_get(card, { 'config', 'center', 'name' })
    _card.set = safe_get(card, { 'ability', 'set' })

    if safe_get(card, { 'config', 'card' }) and next(card.config.card) ~= nil then
        _card.suit = safe_get(card, { 'base', 'suit' })
        _card.value = safe_get(card, { 'base', 'value' })
        _card.card_key = safe_get(card, { 'config', 'card_key' })
    end

    if card.config and card.config.center then
        _card.center_key = card.config.center_key
        _card.cost = card.cost
        _card.sell_cost = card.sell_cost
    end

    if card.ability then
        _card.debuff = card.debuff
        _card.ability_name = card.ability.name
        _card.mult = card.ability.mult
        _card.x_mult = card.ability.x_mult
        _card.chips = card.ability.bonus
        _card.perma_bonus = card.ability.perma_bonus
        if card.ability.extra and type(card.ability.extra) == 'table' then
            _card.extra = {}
            for k, v in pairs(card.ability.extra) do
                if type(v) ~= 'function' then
                    _card.extra[k] = v
                end
            end
        end
    end

    if card.edition then
        _card.edition = card.edition
    end

    if card.seal then
        _card.seal = card.seal
    end
    
    if card.sticker then
        _card.sticker = card.sticker
    end
    if card.sticker_run then
        _card.sticker_run = card.sticker_run
    end

    _card.eternal = safe_get(card, { 'ability', 'eternal' })
    _card.perishable = safe_get(card, { 'ability', 'perishable' })
    if _card.perishable then
        _card.perish_tally = safe_get(card, { 'ability', 'perish_tally' })
    end
    _card.rental = safe_get(card, { 'ability', 'rental' })
    _card.pinned = card.pinned

    return _card
end

function Utils.getDeckData()
    local _deck = { cards = {}, discards = {} }

    if G and G.deck and G.deck.cards then
        for i = 1, #G.deck.cards do
            _deck.cards[i] = Utils.getCardData(G.deck.cards[i])
        end
    end

    if G and G.discard and G.discard.cards then
        for i = 1, #G.discard.cards do
            _deck.discards[i] = Utils.getCardData(G.discard.cards[i])
        end
    end

    return _deck
end

function Utils.getHandData()
    local _hand = {}

    if G and G.hand and G.hand.cards then
        for i = 1, #G.hand.cards do
            local _card = Utils.getCardData(G.hand.cards[i])
            _hand[i] = _card
        end
    end

    return _hand
end

function Utils.getJokersData()
    local _jokers = {}

    if G and G.jokers and G.jokers.cards then
        for i = 1, #G.jokers.cards do
            local _card = Utils.getCardData(G.jokers.cards[i])
            _jokers[i] = _card
        end
    end

    return _jokers
end

function Utils.getConsumablesData()
    local _consumables = {}

    if G and G.consumables and G.consumables.cards then
        for i = 1, #G.consumables.cards do
            local _card = Utils.getCardData(G.consumables.cards[i])
            _consumables[i] = _card
        end
    end

    return _consumables
end

function Utils.getBlindData()
    local _blinds = { current = nil, small = nil, big = nil, boss = nil, ondeck = nil }

    local function get_blind_info(key)
        if not key or not G.P_BLINDS[key] then return nil end
        local proto = G.P_BLINDS[key]
        local info = {
            name = proto.name,
            dollars = proto.dollars,
            mult = proto.mult,
            debuff = proto.debuff,
            boss = proto.boss,
            key = key
        }
        if G.GAME and G.GAME.round_resets and get_blind_amount and G.GAME.starting_params then
             info.chips = get_blind_amount(G.GAME.round_resets.ante) * info.mult * G.GAME.starting_params.ante_scaling
        end
        return info
    end

    if G and G.GAME and G.GAME.blind and G.GAME.blind.config and G.GAME.blind.config.blind then
        local b = G.GAME.blind
        _blinds.current = {
            name = b.name,
            dollars = b.dollars,
            mult = b.mult,
            chips = b.chips,
            debuff = b.debuff,
            boss = b.boss,
            disabled = b.disabled,
            key = b.config.blind.key
        }
    end

    if G and G.GAME and G.GAME.round_resets and G.GAME.round_resets.blind_choices then
        local choices = G.GAME.round_resets.blind_choices
        _blinds.small = get_blind_info(choices.Small)
        _blinds.big = get_blind_info(choices.Big)
        _blinds.boss = get_blind_info(choices.Boss)
    end
    
    if G and G.GAME then
        _blinds.ondeck_key = G.GAME.blind_on_deck
        _blinds.ondeck = get_blind_info(G.GAME.blind_on_deck)
    end

    return _blinds
end

function Utils.getAnteData()
    local _ante = {}
    if G and G.GAME then
        if G.GAME.round_resets then
            _ante.ante = G.GAME.round_resets.ante
            _ante.blind_ante = G.GAME.round_resets.blind_ante
            _ante.blind_states = G.GAME.round_resets.blind_states
        end
        _ante.win_ante = G.GAME.win_ante
    end
    _ante.blinds = Utils.getBlindData()

    return _ante
end

function Utils.getBackData()
    local _back = {}
    if G and G.GAME and G.GAME.selected_back_key and G.P_CENTERS[G.GAME.selected_back_key] then
        local back_proto = G.P_CENTERS[G.GAME.selected_back_key]
        _back.name = back_proto.name
        _back.key = G.GAME.selected_back_key
        _back.config = back_proto.config
    end
    return _back
end

function Utils.getShopData()
    local _shop = { jokers = {}, boosters = {}, vouchers = {}, tarots = {}, planets = {}, playing_cards = {} }
    if not G or not G.shop then return _shop end
    
    _shop.reroll_cost = safe_get(G, {'GAME', 'current_round', 'reroll_cost'})
    _shop.free_rerolls = safe_get(G, {'GAME', 'current_round', 'free_rerolls'})

    local shop_areas = {
        {area = 'shop_jokers', key = 'jokers'},
        {area = 'shop_booster', key = 'boosters'},
        {area = 'shop_vouchers', key = 'vouchers'},
        {area = 'shop_tarot', key = 'tarots'},
        {area = 'shop_planet', key = 'planets'},
        {area = 'shop_standard', key = 'playing_cards'}
    }

    for _, v in ipairs(shop_areas) do
        if G[v.area] and G[v.area].cards then
            for i = 1, #G[v.area].cards do
                _shop[v.key][i] = Utils.getCardData(G[v.area].cards[i])
            end
        end
    end

    return _shop
end

function Utils.getHandScoreData()
    local _handscores = {}
    if G and G.GAME and G.GAME.hands then
        for hand, data in pairs(G.GAME.hands) do
            if data.visible then
                _handscores[hand] = {
                    level = data.level,
                    chips = data.chips,
                    mult = data.mult
                }
            end
        end
    end
    return _handscores
end

function Utils.getTagsData()
    local _tags = {}
    if G and G.GAME and G.GAME.tags then
        for i = 1, #G.GAME.tags do
            local tag = G.GAME.tags[i]
            _tags[i] = {
                key = tag.key,
                name = tag.name,
                config = tag.config
            }
        end
    end
    return _tags
end

function Utils.getRoundData()
    local _current_round = {}

    if G and G.GAME and G.GAME.current_round then
        _current_round.discards_left = G.GAME.current_round.discards_left
        _current_round.hands_left = G.GAME.current_round.hands_left
        _current_round.reroll_cost = G.GAME.current_round.reroll_cost
        _current_round.free_rerolls = G.GAME.current_round.free_rerolls
        _current_round.hands_played_this_round = G.GAME.current_round.hands_played
        _current_round.discards_used_this_round = G.GAME.current_round.discards_used
        _current_round.blind_on_deck = G.GAME.blind_on_deck
    end

    return _current_round
end

function Utils.getGameData()
    local _game = {}

    if G and G.GAME then
        _game.state = G.STATE
        _game.hands_played = G.GAME.hands_played
        _game.skips = G.GAME.skips
        _game.round = G.GAME.round
        _game.ante = safe_get(G, {'GAME', 'round_resets', 'ante'})
        _game.discount_percent = G.GAME.discount_percent
        _game.interest_cap = G.GAME.interest_cap
        _game.interest_amount = G.GAME.interest_amount
        _game.inflation = G.GAME.inflation
        _game.dollars = G.GAME.dollars
        _game.max_jokers = G.GAME.max_jokers
        _game.max_consumables = safe_get(G, {'consumables', 'config', 'card_limit'})
        _game.bankrupt_at = G.GAME.bankrupt_at
        _game.chips = G.GAME.chips
        _game.win_streak = safe_get(G, {'PROFILES', G.SETTINGS.profile, 'high_scores', 'win_streak', 'amt'})
        _game.current_streak = safe_get(G, {'PROFILES', G.SETTINGS.profile, 'high_scores', 'current_streak', 'amt'})
        _game.stake = G.GAME.stake
    end

    return _game
end

function Utils.getGamestate()
    if not G then return nil end
    local _gamestate = {}

    _gamestate.state = G.STATE
    _gamestate.waiting_for = G.waitingFor
    
    _gamestate.game = Utils.getGameData()
    _gamestate.round = Utils.getRoundData()
    _gamestate.ante = Utils.getAnteData()
    _gamestate.hand = Utils.getHandData()
    _gamestate.deck = Utils.getDeckData()
    _gamestate.jokers = Utils.getJokersData()
    _gamestate.consumables = Utils.getConsumablesData()
    _gamestate.shop = Utils.getShopData()
    _gamestate.hand_scores = Utils.getHandScoreData()
    _gamestate.tags = Utils.getTagsData()
    _gamestate.back = Utils.getBackData()

    return _gamestate
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

        local _actiontable = {}
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

    local _seen = {}
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