Actions = {}
Actions.executing = false
-- Helper function to find and push the correct button on a card (buy, use, sell, etc.)
-- This is executed immediately within an event, not queued.
local function execute_use_card(card)
    if not card then return end

    local _use_button = card.children.use_button and card.children.use_button.definition
    if _use_button and _use_button.config.button == nil then
        local _node_index = card.ability.consumeable and 2 or 1
        _use_button = _use_button.nodes[_node_index]

        if card.area and card.area.config.type == 'joker' then
            _use_button = card.children.use_button.definition.nodes[1].nodes[1].nodes[1].nodes[1]
        end
    end

    local _buy_and_use_button = card.children.buy_and_use_button and card.children.buy_and_use_button.definition
    local _buy_button = card.children.buy_button and card.children.buy_button.definition

    local button_to_push = _use_button or _buy_and_use_button or _buy_button

    if button_to_push and button_to_push.config.button then
        G.FUNCS[button_to_push.config.button](button_to_push)
    end
end

function Actions.done() 
    sendDebugMessage("Action done")
    Actions.executing = false
end

-- Action to play selected cards from hand
function Actions.play_hand(cards_to_play)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if cards_to_play then
                for i = 1, #cards_to_play do
                    local card = G.hand.cards[cards_to_play[i]]
                    if card then card:click() end
                end
            else 
                return false
            end
            return true
        end
    }))
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local play_button = UIBox:get_UIE_by_ID('play_button', G.buttons.UIRoot)
            if play_button and play_button.config.button then
                G.FUNCS[play_button.config.button](play_button)
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to discard selected cards from hand
function Actions.discard_hand(cards_to_discard)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if cards_to_discard then
                for i = 1, #cards_to_discard do
                    local card = G.hand.cards[cards_to_discard[i]]
                    if card then card:click() end
                end
            end
            local discard_button = UIBox:get_UIE_by_ID('discard_button', G.buttons.UIRoot)
            if discard_button and discard_button.config.button then
                G.FUNCS[discard_button.config.button](discard_button)
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to select the upcoming blind
function Actions.select_blind()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local blind_on_deck = G.GAME.blind_on_deck
            if blind_on_deck then
                local blind_obj = G.blind_select_opts[string.lower(blind_on_deck)]
                if blind_obj then
                    local select_button = blind_obj:get_UIE_by_ID('select_blind_button')
                    if select_button and select_button.config.button then
                        G.FUNCS[select_button.config.button](select_button)
                    end
                end
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to skip the upcoming blind
function Actions.skip_blind()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            G.currently_executing_action = true
            local blind_on_deck = G.GAME.blind_on_deck
            if blind_on_deck then
                local blind_obj = G.blind_select_opts[string.lower(blind_on_deck)]
                if blind_obj then
                    local tag_button = blind_obj:get_UIE_by_ID('tag_'..blind_on_deck)
                    if tag_button and tag_button.children[2] and tag_button.children[2].config.button then
                        local skip_button = tag_button.children[2]
                        G.FUNCS[skip_button.config.button](skip_button)
                    end
                end
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to buy a card (Joker) from the shop
function Actions.buy_card(card_index)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.shop_jokers and G.shop_jokers.cards[card_index] then
                local card = G.shop_jokers.cards[card_index]
                card:click()
                execute_use_card(card)
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to buy a voucher from the shop
function Actions.buy_voucher(voucher_index)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.shop_vouchers and G.shop_vouchers.cards[voucher_index] then
                local card = G.shop_vouchers.cards[voucher_index]
                card:click()
                execute_use_card(card)
            else 
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to buy a booster pack from the shop
function Actions.buy_booster(booster_index)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.shop_booster and G.shop_booster.cards[booster_index] then
                local card = G.shop_booster.cards[booster_index]
                card:click()
                execute_use_card(card)
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to reroll the shop
function Actions.reroll_shop()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local reroll_button = UIBox:get_UIE_by_ID('reroll_button', G.buttons.UIRoot)
            if reroll_button and reroll_button.config.button then
                 G.FUNCS[reroll_button.config.button](reroll_button)
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to leave the shop and start the next round
function Actions.end_shop()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local next_round_button = UIBox:get_UIE_by_ID('next_round_button', G.buttons.UIRoot)
            if not next_round_button then
                -- Fallback based on middleware logic
                next_round_button = UIBox:get_UIE_by_ID('toggle_shop_button', G.buttons.UIRoot)
            end
            if next_round_button and next_round_button.config.button then
                 G.FUNCS[next_round_button.config.button](next_round_button)
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to skip a booster pack
function Actions.skip_booster_pack()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local skip_button = UIBox:get_UIE_by_ID('skip_button', G.pack_cards)
            if skip_button and skip_button.config.button == 'skip_booster' then
                G.FUNCS[skip_button.config.button](skip_button)
            else 
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to select a card from a booster pack
function Actions.select_booster_card(booster_card_index, hand_card_indices)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if hand_card_indices then
                for i = 1, #hand_card_indices do
                    local card = G.hand.cards[hand_card_indices[i]]
                    if card then card:click() end
                end
            end

            if G.pack_cards and G.pack_cards.cards[booster_card_index] then
                local booster_card = G.pack_cards.cards[booster_card_index]
                booster_card:click()
                execute_use_card(booster_card)
            else 
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to sell jokers
function Actions.sell_joker(joker_indices)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.jokers and G.jokers.cards and joker_indices then
                for i = 1, #joker_indices do
                    local card = G.jokers.cards[joker_indices[i]]
                    if card then
                        card:click()
                        execute_use_card(card)
                    end
                end
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to use a consumable card
function Actions.use_consumable(consumable_indices, hand_card_indices)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.consumeables and G.consumeables.cards and consumable_indices then
                if hand_card_indices then
                    for i = 1, #hand_card_indices do
                        local card = G.hand.cards[hand_card_indices[i]]
                        if card then card:click() end
                    end
                end
                for i = 1, #consumable_indices do
                    local card = G.consumeables.cards[consumable_indices[i]]
                    if card then
                        card:click()
                        execute_use_card(card)
                    end
                end
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to sell a consumable card
function Actions.sell_consumable(consumable_indices)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.consumeables and G.consumeables.cards and consumable_indices then
                for i = 1, #consumable_indices do
                    local card = G.consumeables.cards[consumable_indices[i]]
                    if card then
                        card:click()
                        execute_use_card(card)
                    end
                end
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to rearrange cards in hand
function Actions.rearrange_hand(order)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.hand and G.hand.cards and order and #order == #G.hand.cards then
                local new_hand = {}
                for i = 1, #order do
                    new_hand[i] = G.hand.cards[order[i]]
                end
                G.hand.cards = new_hand
                G.hand:set_ranks()
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to rearrange jokers
function Actions.rearrange_jokers(order)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.jokers and G.jokers.cards and order and #order == #G.jokers.cards then
                local new_jokers = {}
                for i = 1, #order do
                    new_jokers[i] = G.jokers.cards[order[i]]
                end
                G.jokers.cards = new_jokers
                G.jokers:set_ranks()
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to rearrange consumables
function Actions.rearrange_consumables(order)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            if G.consumeables and G.consumeables.cards and order and #order == #G.consumeables.cards then
                local new_consumables = {}
                for i = 1, #order do
                    new_consumables[i] = G.consumeables.cards[order[i]]
                end
                G.consumeables.cards = new_consumables
                G.consumeables:set_ranks()
            else
                return false
            end
            Actions.done()
            return true
        end
    }))
end

-- Action to start a new run
function Actions.start_run(stake, deck, seed, challenge)
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            local params = {
                stake = stake or 1,
                seed = seed,
                challenge = challenge
            }
            local deck_name = deck or "Red Deck"

            for k, v in pairs(G.P_CENTER_POOLS.Back) do
                if v.name == deck_name then
                    G.GAME.selected_back:change_to(v)
                end
            end

            if params.challenge then
                for i = 1, #G.CHALLENGES do
                    if G.CHALLENGES[i].name == params.challenge then
                        params.challenge = G.CHALLENGES[i]
                        break
                    end
                end
            end
            G.FUNCS.start_run(nil, params)
            Actions.done()
            return true
        end
    }))
end

-- Action to return to the main menu
function Actions.return_to_menu()
    G.E_MANAGER:add_event(Event({
        trigger = 'after', delay = 0.5, blocking = true,
        func = function()
            G.FUNCS.go_to_menu()
            Actions.done()
            return true
        end
    }))
end

return Actions