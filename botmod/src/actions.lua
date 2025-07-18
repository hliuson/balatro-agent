Actions = {}
Actions.executing = false
Actions.Buttons = {
    cash_out_button = nil,
    next_round_button = nil,
}
-- Helper function to find and push the correct button on a card (buy, use, sell, etc.)
local function execute_use_card(card)
    if not card then return end

    local _use_button = card.children.use_button and card.children.use_button.definition
    if _use_button and _use_button.config and _use_button.config.button == nil then
        local _node_index = card.ability.consumeable and 2 or 1
        _use_button = _use_button.nodes[_node_index]

        if card.area and card.area.config.type == 'joker' then
            _use_button = card.children.use_button.definition.nodes[1].nodes[1].nodes[1].nodes[1]
        end
    end

    local _buy_and_use_button = card.children.buy_and_use_button and card.children.buy_and_use_button.definition
    local _buy_button = card.children.buy_button and card.children.buy_button.definition

    local button_to_push = _use_button or _buy_and_use_button or _buy_button

    if button_to_push and button_to_push.config and button_to_push.config.button then
        G.FUNCS[button_to_push.config.button](button_to_push)
    end
end

function Actions.done()
    Actions.executing = false
end

-- SAFE ACTION WRAPPER
-- This function wraps all actions in a safe, condition-based event
local function safe_action(action_func)
    local tries = 0
    
    G.E_MANAGER:add_event(Event({
        trigger = 'immediate',
        blocking = false,
        blockable = true,
        func = function()
            -- Essential safety checks from the botting guide
            if G.CONTROLLER.locked or not G.STATE_COMPLETE then
                return false -- Wait until the game is ready
            end

            -- Animation completion check
            for k, v in pairs(G.I.CARD) do
                if v.T.x ~= v.VT.x or v.T.y ~= v.VT.y then
                    return false -- Cards are still moving
                end
            end
            -- Execute the core action logic
            local success = action_func()
            if success then
                Actions.done()
            else
                return false -- Continue waiting for the next frame
            end
            return true --even if the action is not successful, we return true to yield position in queue
        end
    }))
end

-- Action to select/deselect a hand card
function Actions.select_hand_card(card_index)
    safe_action(function()
        if G.hand and G.hand.cards and card_index >= 1 and card_index <= #G.hand.cards then
            local card = G.hand.cards[card_index]
            if card then
                card:click()  -- Game handles selection state automatically
            end
            return true
        end
        return false
    end)
end

-- Action to clear all hand card selections
function Actions.clear_hand_selection()
    safe_action(function()
        if G.hand and G.hand.cards then
            for i = 1, #G.hand.cards do
                local card = G.hand.cards[i]
                if card and card.highlighted then
                    card:click() -- Deselect if currently selected
                end
            end
        end
        return true
    end)
end

-- Action to play selected cards from hand
function Actions.play_selected()
    safe_action(function()
        if not G.buttons or not G.buttons.UIRoot then
            return false -- Wait for buttons to be ready
        end
        local play_button = UIBox:get_UIE_by_ID('play_button', G.buttons.UIRoot)
        if play_button and play_button.config and play_button.config.button then
            G.FUNCS[play_button.config.button](play_button)
            return true -- Action complete
        end
        return false
    end)
end

-- Action to discard selected cards from hand
function Actions.discard_selected()
    safe_action(function()
        if not G.buttons or not G.buttons.UIRoot then
            return false -- Wait for buttons to be ready
        end
        local discard_button = UIBox:get_UIE_by_ID('discard_button', G.buttons.UIRoot)
        if discard_button and discard_button.config and discard_button.config.button then
            G.FUNCS[discard_button.config.button](discard_button)
            return true -- Action complete
        end
        return false
    end)
end

-- Action to select the upcoming blind
function Actions.select_blind()
    safe_action(function()
        local blind_on_deck = G.GAME.blind_on_deck
        if blind_on_deck then
            local blind_obj = G.blind_select_opts[string.lower(blind_on_deck)]
            if blind_obj then
                local select_button = blind_obj:get_UIE_by_ID('select_blind_button')
                if select_button and select_button.config and select_button.config.button then
                    G.FUNCS[select_button.config.button](select_button)
                    return true -- Action complete
                end
            end
        end
        return false
    end)
end

-- Action to skip the upcoming blind
function Actions.skip_blind()
    safe_action(function()
        local blind_on_deck = G.GAME.blind_on_deck
        if blind_on_deck then
            local blind_obj = G.blind_select_opts[string.lower(blind_on_deck)]
            if blind_obj then
                local tag_button = blind_obj:get_UIE_by_ID('tag_'..blind_on_deck)
                if tag_button and tag_button.children[2] and tag_button.children[2].config and tag_button.children[2].config.button then
                    local skip_button = tag_button.children[2]
                    G.FUNCS[skip_button.config.button](skip_button)
                    return true -- Action complete
                end
            end
        end
        return false
    end)
end

-- Action to buy a card (Joker) from the shop
function Actions.buy_card(card_index)
    safe_action(function()
        if G.shop_jokers and G.shop_jokers.cards[card_index] then
            local card = G.shop_jokers.cards[card_index]
            card:click()
            execute_use_card(card)
            return true
        end
        return false
    end)
end

-- Action to buy a voucher from the shop
function Actions.buy_voucher(voucher_index)
    safe_action(function()
        if G.shop_vouchers and G.shop_vouchers.cards[voucher_index] then
            local card = G.shop_vouchers.cards[voucher_index]
            card:click()
            execute_use_card(card)
        end
        return false
    end)
end

-- Action to buy a booster pack from the shop
function Actions.buy_booster(booster_index)
    safe_action(function()
        if G.shop_booster and G.shop_booster.cards[booster_index] then
            local card = G.shop_booster.cards[booster_index]
            card:click()
            execute_use_card(card)
        end
        return false
    end)
end

-- Action to reroll the shop
function Actions.reroll_shop()
    safe_action(function()
        if not G.buttons or not G.buttons.UIRoot then
            return false -- Wait for buttons to be ready
        end
        local reroll_button = UIBox:get_UIE_by_ID('reroll_button', G.buttons.UIRoot)
        if reroll_button and reroll_button.config and reroll_button.config.button then
             G.FUNCS[reroll_button.config.button](reroll_button)
        end
        return false
    end)
end

-- Action to leave the shop and start the next round
function Actions.end_shop()
    safe_action(function()
        --local next_round_button = Actions.Buttons.next_round_button
        --if next_round_button and next_round_button.config and next_round_button.config.button then
        --    G.FUNCS[next_round_button.config.button](next_round_button)
        --    Actions.Buttons.next_round_button = nil -- Clear the button after use
        --    return true -- Action complete
        --end
        G.FUNCS.toggle_shop(nil) 
        return true -- Action complete
    end)
end

-- Action to skip a booster pack
function Actions.skip_booster_pack()
    safe_action(function()
        local skip_button = UIBox:get_UIE_by_ID('skip_button', G.pack_cards)
        if skip_button and skip_button.config and skip_button.config.button == 'skip_booster' then
            G.FUNCS[skip_button.config.button](skip_button)
            return true
        end
        return false
    end)
end

-- Action to select a card from a booster pack
function Actions.select_booster_card(booster_card_index)
    safe_action(function()
        if G.pack_cards and G.pack_cards.cards[booster_card_index] then
            local booster_card = G.pack_cards.cards[booster_card_index]
            booster_card:click()
            execute_use_card(booster_card)
            return true
        end
        return false
    end)
end

-- Action to sell jokers
function Actions.sell_joker(joker_indices)
    safe_action(function()
        if G.jokers and G.jokers.cards and joker_indices then
            for i = 1, #joker_indices do
                local card = G.jokers.cards[joker_indices[i]]
                if card then
                    card:click()
                    execute_use_card(card)
                end
            end
            return true
        end
        return false
    end)
end

-- Action to use a consumable card
function Actions.use_consumable(consumable_index)
    safe_action(function()
        if G.consumeables and G.consumeables.cards and consumable_index then
            local card = G.consumeables.cards[consumable_index]
            if card then
                card:click()
                execute_use_card(card)
                return true
            end
        end
        return false
    end)
end

-- Action to sell a consumable card
function Actions.sell_consumable(consumable_indices)
    safe_action(function()
        if G.consumeables and G.consumeables.cards and consumable_indices then
            for i = 1, #consumable_indices do
                local card = G.consumeables.cards[consumable_indices[i]]
                if card then
                    card:click()
                    execute_use_card(card)
                end
            end
            return true
        end
        return false
    end)
end

--[[
UNSAFE ACTIONS - These directly manipulate game state and are not recommended.
They are left here for reference but should be used with extreme caution.
It is recommended to achieve these outcomes through sequences of safe, UI-driven actions.
]]
function Actions.rearrange_hand(order)
    sendDebugMessage("WARNING: Unsafe action 'rearrange_hand' used.")
    safe_action(function()
        if G.hand and G.hand.cards and order and #order == #G.hand.cards then
            local new_hand = {}
            for i = 1, #order do
                new_hand[i] = G.hand.cards[order[i]]
            end
            G.hand.cards = new_hand
            G.hand:set_ranks()
            return true
        end
        return false
    end)
end

function Actions.rearrange_jokers(order)
    sendDebugMessage("WARNING: Unsafe action 'rearrange_jokers' used.")
    safe_action(function()
        if G.jokers and G.jokers.cards and order and #order == #G.jokers.cards then
            local new_jokers = {}
            for i = 1, #order do
                new_jokers[i] = G.jokers.cards[order[i]]
            end
            G.jokers.cards = new_jokers
            G.jokers:set_ranks()
            return true
        end
        return false
    end)
end

function Actions.rearrange_consumables(order)
    sendDebugMessage("WARNING: Unsafe action 'rearrange_consumables' used.")
    safe_action(function()
        if G.consumeables and G.consumeables.cards and order and #order == #G.consumeables.cards then
            local new_consumables = {}
            for i = 1, #order do
                new_consumables[i] = G.consumeables.cards[order[i]]
            end
            G.consumeables.cards = new_consumables
            G.consumeables:set_ranks()
            return true
        end
        return false
    end)
end

-- Action to start a new run safely
function Actions.start_run(stake, deck, seed, challenge)
    G.FUNCS.start_run(nil, {
                stake = stake or 1,
                seed = seed,
                challenge = challenge,
                deck = deck or "Red Deck",
                skip_menu = true -- Recommended for bots
    })
    Actions.done()
end
-- Action to return to the main menu
function Actions.return_to_menu()
    safe_action(function()
        G.FUNCS.go_to_menu()
        return true
    end)
end

function Actions.pass()
    Actions.done()
end

function Actions.cash_out()
    local stage = 1
    safe_action(function()
        if stage == 1 then
            local cash_out_button = Actions.Buttons.cash_out_button
            if cash_out_button and cash_out_button.config and cash_out_button.config.button then
                G.FUNCS[cash_out_button.config.button](cash_out_button)
                Actions.Buttons.cash_out_button = nil -- Clear the button after use
                stage = 2
                return false -- Action complete
            end
        elseif stage == 2 then
            --wait for shop to be ready before calling done
            if G.shop_jokers then
                Actions.done()
                return true -- Action complete
            end
        end
        
    end)
end

G.CONTROLLER.snap_to = Hook.addcallback(G.CONTROLLER.snap_to, function(...)
    local _self = ...
    if _self and _self.snap_cursor_to.node and _self.snap_cursor_to.node.config and _self.snap_cursor_to.node.config.button then
        
        local _button = _self.snap_cursor_to.node
        local _buttonfunc = _self.snap_cursor_to.node.config.button

        --if _buttonfunc == 'select_blind' and G.STATE == G.STATES.BLIND_SELECT then
        --    Middleware.c_select_blind()
        if _buttonfunc == 'cash_out' then
            Actions.Buttons.cash_out_button = _button
        elseif _buttonfunc == 'toggle_shop' and G.shop ~= nil then -- 'next_round_button'
            Actions.Buttons.next_round_button = _button
        end
        
        --    firewhenready(function()
        --        return G.shop ~= nil and G.STATE_COMPLETE and G.STATE == G.STATES.SHOP
        --    end, Middleware.c_shop)
        --end
    end
end)

return Actions