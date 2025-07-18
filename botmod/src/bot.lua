
Bot = { }

-- DO NOT TOUCH
Bot.ACTIONS = {
    SELECT_BLIND = 1,
    SKIP_BLIND = 2,
    SELECT_HAND_CARD = 3,
    CLEAR_HAND_SELECTION = 4,
    PLAY_SELECTED = 5,
    DISCARD_SELECTED = 6,
    END_SHOP = 7,
    REROLL_SHOP = 8,
    BUY_CARD = 9,
    BUY_VOUCHER = 10,
    BUY_BOOSTER = 11,
    SELECT_BOOSTER_CARD = 12,
    SKIP_BOOSTER_PACK = 13,
    SELL_JOKER = 14,
    USE_CONSUMABLE = 15,
    SELL_CONSUMABLE = 16,
    REARRANGE_JOKERS = 17,
    REARRANGE_CONSUMABLES = 18,
    REARRANGE_HAND = 19,
    PASS = 20,
    START_RUN = 21,
    RETURN_TO_MENU = 22,
    CASH_OUT = 23
}

Bot.ACTIONPARAMS = { }
Bot.ACTIONPARAMS[Bot.ACTIONS.SELECT_BLIND] = {
    num_args = 1,
    isvalid = function(action)
        if G.STATE == G.STATES.BLIND_SELECT then return true end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SKIP_BLIND] = {
    num_args = 1,
    isvalid = function(action)
        if G.STATE == G.STATES.BLIND_SELECT then return true end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SELECT_HAND_CARD] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.hand and G.hand.cards and #action == 2 and
            type(action[2]) == "table" and action[2][1] >= 1 and action[2][1] <= #G.hand.cards then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.CLEAR_HAND_SELECTION] = {
    num_args = 1,
    isvalid = function(action)
        return true
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.PLAY_SELECTED] = {
    num_args = 1,
    isvalid = function(action)
        if G and G.GAME and G.GAME.current_round and
            G.GAME.current_round.hands_left > 0 and
            G.hand and G.hand.cards then
            -- Check if any cards are currently highlighted/selected
            for i = 1, #G.hand.cards do
                if G.hand.cards[i] and G.hand.cards[i].highlighted then
                    return true
                end
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.DISCARD_SELECTED] = {
    num_args = 1,
    isvalid = function(action)
        if G and G.GAME and G.GAME.current_round and
            G.GAME.current_round.discards_left > 0 and
            G.hand and G.hand.cards then
            -- Check if any cards are currently highlighted/selected
            for i = 1, #G.hand.cards do
                if G.hand.cards[i] and G.hand.cards[i].highlighted then
                    return true
                end
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.END_SHOP] = {
    num_args = 1,
    isvalid = function(action)
        if G and G.STATE == G.STATES.SHOP then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.REROLL_SHOP] = {
    num_args = 1,
    isvalid = function(action)
        if G and G.STATE == G.STATES.SHOP and (G.GAME.dollars - G.GAME.bankrupt_at - G.GAME.current_round.reroll_cost >= 0) then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.BUY_CARD] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.STATE == G.STATES.SHOP and #action == 2 and #action[2] == 1 and
        G.shop_jokers and G.shop_jokers.cards and #G.shop_jokers.cards >= action[2][1] and
        (G.GAME.dollars - G.GAME.bankrupt_at - G.shop_jokers.cards[action[2][1]].cost >= 0) then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.BUY_VOUCHER] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.STATE == G.STATES.SHOP and #action == 2 and #action[2] == 1 and
        G.shop_vouchers and G.shop_vouchers.cards and #G.shop_vouchers.cards >= action[2][1] and
        (G.GAME.dollars - G.GAME.bankrupt_at - G.shop_vouchers.cards[action[2][1]].cost >= 0) then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.BUY_BOOSTER] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.STATE == G.STATES.SHOP and #action == 2 and #action[2] == 1 and
        G.shop_booster and G.shop_booster.cards and #G.shop_booster.cards >= action[2][1] and
        (G.GAME.dollars - G.GAME.bankrupt_at - G.shop_booster.cards[action[2][1]].cost >= 0) then
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SELECT_BOOSTER_CARD] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.hand and G.pack_cards and
        G.hand.cards and G.pack_cards.cards and 
        (G.STATE == G.STATES.TAROT_PACK or
        G.STATE == G.STATES.PLANET_PACK or
        G.STATE == G.STATES.SPECTRAL_PACK or
        G.STATE == G.STATES.STANDARD_PACK or
        G.STATE == G.STATES.BUFFOON_PACK) and
        type(action[2]) == "table" and action[2][1] >= 1 and action[2][1] <= #G.pack_cards.cards then
            local booster_card = G.pack_cards.cards[action[2][1]]
            if booster_card and booster_card.ability and booster_card.ability.consumeable and
            booster_card.ability.consumeable.max_highlighted then
                -- Count currently selected hand cards
                local selected_count = 0
                for i = 1, #G.hand.cards do
                    if G.hand.cards[i] and G.hand.cards[i].highlighted then
                        selected_count = selected_count + 1
                    end
                end
                return selected_count <= booster_card.ability.consumeable.max_highlighted
            end
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SKIP_BOOSTER_PACK] = {
    num_args = 1,
    isvalid = function(action)
        if G.pack_cards and G.pack_cards.cards and G.pack_cards.cards[1] and 
        (G.STATE == G.STATES.PLANET_PACK or 
        G.STATE == G.STATES.STANDARD_PACK or 
        G.STATE == G.STATES.BUFFOON_PACK or 
        (G.hand and G.hand.cards[1]))-- and
        -- Middleware.BUTTONS.SKIP_PACK ~= nil and
        -- Middleware.BUTTONS.SKIP_PACK.config.button == 'skip_booster'
        then 
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SELL_JOKER] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.jokers and G.jokers.cards then
            if not action[2] then return true end

            if Utils.isTableInRange(action[2], 1, #G.jokers.cards) and
            not G.jokers.cards[action[2][1]].ability.eternal then
                return true
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.USE_CONSUMABLE] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.consumeables and G.consumeables.cards and
            type(action[2]) == "table" and action[2][1] >= 1 and action[2][1] <= #G.consumeables.cards then
            local consumable = G.consumeables.cards[action[2][1]]
            if consumable and consumable.ability and consumable.ability.consumeable and
            consumable.ability.consumeable.max_highlighted then
                -- Count currently selected hand cards
                local selected_count = 0
                if G.hand and G.hand.cards then
                    for i = 1, #G.hand.cards do
                        if G.hand.cards[i] and G.hand.cards[i].highlighted then
                            selected_count = selected_count + 1
                        end
                    end
                end
                return selected_count <= consumable.ability.consumeable.max_highlighted
            end
            return true
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.SELL_CONSUMABLE] = {
    num_args = 2,
    isvalid = function(action)
        -- TODO implement this
        return true
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.REARRANGE_JOKERS] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.jokers and G.jokers.cards then
            if not action[2] then return true end

            if Utils.isTableUnique(action[2]) and
            Utils.isTableInRange(action[2], 1, #G.jokers.cards) and
            #action[2] == #G.jokers.cards then
                return true
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.REARRANGE_CONSUMABLES] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.consumeables and G.consumeables.cards then
            if not action[2] then return true end

            if Utils.isTableUnique(action[2]) and
            Utils.isTableInRange(action[2], 1, #G.consumeables.cards) and
            #action[2] == #G.consumeables.cards then
                return true
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.REARRANGE_HAND] = {
    num_args = 2,
    isvalid = function(action)
        if G and G.hand and G.hand.cards then
            if not action[2] then return true end

            if Utils.isTableUnique(action[2]) and
            Utils.isTableInRange(action[2], 1, #G.hand.cards) and
            #action[2] == #G.hand.cards then
                return true
            end
        end
        return false
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.PASS] = {
    num_args = 1,
    isvalid = function(action)
        return true
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.START_RUN] = {
    num_args = 5,
    isvalid = function(action)
        if G and G.STATE == G.STATES.MENU then
            return true
        end
        return true
    end,
}
Bot.ACTIONPARAMS[Bot.ACTIONS.RETURN_TO_MENU] = { -- always valid
    num_args = 1,
    isvalid = function(action)
        return G.STATE_COMPLETE
    end,
}

Bot.ACTIONPARAMS[Bot.ACTIONS.CASH_OUT] = {
    num_args = 1,
    isvalid = function(action)
        if G and G.STATE == G.STATES.ROUND_EVAL then
            return true
        end
        return false
    end,
}

G.currently_executing_action = false


-- CHANGE ME
Bot.SETTINGS = {
    stake = 1,
    deck = "Plasma Deck",

    -- Keep these nil for random seed
    seed = "1OGB5WO",
    challenge = '',

    -- Time between actions the bot takes (pushing buttons, clicking cards, etc.)
    -- Minimum is 1 frame per action
    action_delay = 0,

    -- Replay actions from file?
    replay = false,

    -- Receive commands from the API?
    api = true,
}




return Bot
