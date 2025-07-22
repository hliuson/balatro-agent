
local socket = require "socket"

local data, msg_or_ip, port_or_nil

BalatrobotAPI = { }
BalatrobotAPI.socket = nil

-- waitingForAction: true when the mod is in a stable, interactive state ready for a command
-- waitingForAction: false when it is busy processing a command or waiting for animations
BalatrobotAPI.waitingForAction = false -- Initially, we are ready for an action: start the game
BalatrobotAPI.action_no = 0

-- State transition tracking to prevent premature ready state
BalatrobotAPI.transition_lock = {
    locked = false,
    expected_state = nil,
    lock_reason = nil
}

-- Shop initialization tracking
BalatrobotAPI.shop_state = {
    waiting_for_shop_init = false,
    waiting_for_reroll = false
}

-- Pack state tracking
BalatrobotAPI.pack_state = {
    waiting_for_pack_state = false,
    target_pack_state = nil,
    waiting_for_hand_draw = false
}

-- Table of states where the API client is expected to provide an action
local INTERACTIVE_STATES = {
    [G.STATES.MENU] = true,
    [G.STATES.SELECTING_HAND] = true,
    [G.STATES.BLIND_SELECT] = true,
    [G.STATES.SHOP] = true,
    [G.STATES.TAROT_PACK] = true,
    [G.STATES.SPECTRAL_PACK] = true,
    [G.STATES.STANDARD_PACK] = true,
    [G.STATES.BUFFOON_PACK] = true,
    [G.STATES.PLANET_PACK] = true,
    [G.STATES.GAME_OVER] = true,
    [G.STATES.ROUND_EVAL] = true,
    [999] = true, -- I'm not sure what this is, but it seems to be a valid state in the game
}


-- Dispatch table to call the appropriate function in actions.lua based on the action enum
local ACTION_DISPATCH = {
    [Bot.ACTIONS.SELECT_BLIND] = function(action) Actions.select_blind() end,
    [Bot.ACTIONS.SKIP_BLIND] = function(action) Actions.skip_blind() end,
    [Bot.ACTIONS.SELECT_HAND_CARD] = function(action) Actions.select_hand_card(action[2][1]) end,
    [Bot.ACTIONS.CLEAR_HAND_SELECTION] = function(action) Actions.clear_hand_selection() end,
    [Bot.ACTIONS.PLAY_SELECTED] = function(action) Actions.play_selected() end,
    [Bot.ACTIONS.DISCARD_SELECTED] = function(action) Actions.discard_selected() end,
    [Bot.ACTIONS.END_SHOP] = function(action) Actions.end_shop() end,
    [Bot.ACTIONS.REROLL_SHOP] = function(action) Actions.reroll_shop() end,
    [Bot.ACTIONS.BUY_CARD] = function(action) Actions.buy_card(action[2][1]) end,
    [Bot.ACTIONS.BUY_VOUCHER] = function(action) Actions.buy_voucher(action[2][1]) end,
    [Bot.ACTIONS.BUY_BOOSTER] = function(action) Actions.buy_booster(action[2][1]) end,
    [Bot.ACTIONS.SELECT_BOOSTER_CARD] = function(action) Actions.select_booster_card(action[2][1]) end,
    [Bot.ACTIONS.SKIP_BOOSTER_PACK] = function(action) Actions.skip_booster_pack() end,
    [Bot.ACTIONS.SELL_JOKER] = function(action) Actions.sell_joker(action[2]) end,
    [Bot.ACTIONS.USE_CONSUMABLE] = function(action) Actions.use_consumable(action[2][1]) end,
    [Bot.ACTIONS.SELL_CONSUMABLE] = function(action) Actions.sell_consumable(action[2]) end,
    [Bot.ACTIONS.REARRANGE_JOKERS] = function(action) Actions.rearrange_jokers(action[2]) end,
    [Bot.ACTIONS.REARRANGE_CONSUMABLES] = function(action) Actions.rearrange_consumables(action[2]) end,
    [Bot.ACTIONS.REARRANGE_HAND] = function(action) Actions.rearrange_hand(action[2]) end,
    [Bot.ACTIONS.START_RUN] = function(action) Actions.start_run(action[2] and action[2][1], action[3] and action[3][1], action[4] and action[4][1], action[5] and action[5][1]) end,
    [Bot.ACTIONS.RETURN_TO_MENU] = function(action) Actions.return_to_menu() end,
    [Bot.ACTIONS.PASS] = function(action) Actions.pass() end,
    [Bot.ACTIONS.CASH_OUT] = function(action) Actions.cash_out() end,
}

-- Packages and sends the current game state to the API client.
function BalatrobotAPI.notifyapiclient()
    local _gamestate = Utils.getGamestate()
    _gamestate.waitingForAction = BalatrobotAPI.waitingForAction
    _gamestate.action_no = BalatrobotAPI.action_no
    local _gamestateJsonString = json.encode(_gamestate)

    if BalatrobotAPI.socket and port_or_nil ~= nil then
        BalatrobotAPI.socket:sendto(string.format("%s", _gamestateJsonString), msg_or_ip, port_or_nil)
    end
end

-- Sends a simple response string to the API client (e.g., for errors).
function BalatrobotAPI.respond(str)
    if BalatrobotAPI.socket and port_or_nil ~= nil then
        response = { }
        response.response = str
        str = json.encode(response)
        BalatrobotAPI.socket:sendto(string.format("%s\n", str), msg_or_ip, port_or_nil)
    end
end

-- State transition lock management
function BalatrobotAPI.setTransitionLock(expected_state, reason)
    BalatrobotAPI.transition_lock.locked = true
    BalatrobotAPI.transition_lock.expected_state = expected_state
    BalatrobotAPI.transition_lock.lock_reason = reason
    sendDebugMessage("Transition lock set: waiting for " .. tostring(expected_state) .. " (" .. tostring(reason) .. ")")
end

function BalatrobotAPI.clearTransitionLock()
    if BalatrobotAPI.transition_lock.locked then
        sendDebugMessage("Transition lock cleared: reached " .. tostring(G.STATE))
        BalatrobotAPI.transition_lock.locked = false
        BalatrobotAPI.transition_lock.expected_state = nil
        BalatrobotAPI.transition_lock.lock_reason = nil
    end
end

function BalatrobotAPI.checkTransitionLock()
    if BalatrobotAPI.transition_lock.locked then
        if G.STATE == BalatrobotAPI.transition_lock.expected_state then
            BalatrobotAPI.clearTransitionLock()
            return false -- Lock cleared, no longer locked
        else
            return true -- Still locked
        end
    end
    return false -- Not locked
end

-- Shop initialization management
function BalatrobotAPI.setShopInitWait()
    BalatrobotAPI.shop_state.waiting_for_shop_init = true
    sendDebugMessage("Cash out initiated - waiting for shop initialization")
end

function BalatrobotAPI.setRerollWait()
    BalatrobotAPI.shop_state.waiting_for_reroll = true
    sendDebugMessage("Reroll initiated - waiting for jokers to be ready")
end

-- Pack state management
function BalatrobotAPI.setPackStateWait(target_state)
    BalatrobotAPI.pack_state.waiting_for_pack_state = true
    BalatrobotAPI.pack_state.target_pack_state = target_state
    
    -- Set hand draw wait for tarot and spectral packs
    if target_state == G.STATES.TAROT_PACK or target_state == G.STATES.SPECTRAL_PACK then
        BalatrobotAPI.pack_state.waiting_for_hand_draw = true
        sendDebugMessage("Pack opened - waiting for state 999, setting to " .. tostring(target_state) .. ", then waiting for hand cards")
    else
        sendDebugMessage("Pack opened - waiting for state 999 then setting to " .. tostring(target_state))
    end
end

function BalatrobotAPI.checkPackStateTransition()
    if BalatrobotAPI.pack_state.waiting_for_pack_state then
        if G.STATE == 999 then
            -- We've reached state 999, now set the target pack state
            G.STATE = BalatrobotAPI.pack_state.target_pack_state
            BalatrobotAPI.pack_state.waiting_for_pack_state = false
            BalatrobotAPI.pack_state.target_pack_state = nil
            sendDebugMessage("Pack state transition complete - set to " .. tostring(G.STATE))
            return false -- No longer waiting
        else
            return true -- Still waiting for state 999
        end
    end
    return false -- Not waiting
end

function BalatrobotAPI.checkPackCardsReady()
    -- Check if we're in a pack state and pack cards are ready for interaction
    local pack_states = {
        [G.STATES.TAROT_PACK] = true,
        [G.STATES.PLANET_PACK] = true,
        [G.STATES.SPECTRAL_PACK] = true,
        [G.STATES.STANDARD_PACK] = true,
        [G.STATES.BUFFOON_PACK] = true
    }
    
    if pack_states[G.STATE] then
        -- Check if pack_cards area exists
        if not G.pack_cards then
            return false -- Pack cards area not created yet
        end
        
        -- Check if pack_cards has the expected number of cards
        if not G.pack_cards.cards or #G.pack_cards.cards < (G.GAME.pack_size or 1) then
            return false -- Not enough cards loaded yet
        end
        return true -- Pack cards are ready
    end
    
    return true -- Not in a pack state, so no pack cards to check
end

function BalatrobotAPI.checkHandDrawComplete()
    -- Check if hand cards are ready after using tarot/spectral cards
    if BalatrobotAPI.pack_state.waiting_for_hand_draw then
        if G.hand and G.hand.cards and #G.hand.cards > 0 then
            BalatrobotAPI.pack_state.waiting_for_hand_draw = false
            sendDebugMessage("Hand draw complete - hand has " .. #G.hand.cards .. " cards")
            return false -- No longer waiting
        else
            return true -- Still waiting for hand cards
        end
    end
    return false -- Not waiting for hand draw
end

function BalatrobotAPI.checkShopInitialization()
    if G.STATE == G.STATES.SHOP then
        -- Check for shop initialization (full shop areas)
        if BalatrobotAPI.shop_state.waiting_for_shop_init then
            local shop_ready = true
            
            -- Check if vouchers area exists
            if not G.shop_vouchers then
                shop_ready = false
            end
            
            -- Check if jokers area exists and has cards
            if not G.shop_jokers or not G.shop_jokers.cards or #G.shop_jokers.cards < 1 then
                shop_ready = false
            end
            
            -- Check if booster area exists
            if not G.shop_booster or not G.shop_booster.cards or #G.shop_booster.cards < 1 then
                shop_ready = false
            end
            
            if shop_ready then
                BalatrobotAPI.shop_state.waiting_for_shop_init = false
                sendDebugMessage("Shop initialization complete")
            else
                return true -- Still waiting for shop init
            end
        end
        
        -- Check for reroll completion (just jokers need to be ready)
        if BalatrobotAPI.shop_state.waiting_for_reroll then
            if G.shop_jokers and G.shop_jokers.cards and #G.shop_jokers.cards >= 1 then
                BalatrobotAPI.shop_state.waiting_for_reroll = false
                sendDebugMessage("Reroll complete - jokers ready")
            else
                return true -- Still waiting for reroll
            end
        end
    end
    
    return false -- Not waiting
end

function BalatrobotAPI.stablestate()
    -- Check if the game state is stable and not in a transition
    local stable = true
    local instability_reasons = {}
    
    if not G.CONTROLLER or G.CONTROLLER.locked then
        stable = false
        table.insert(instability_reasons, "Controller locked or missing")
    end
    if not INTERACTIVE_STATES[G.STATE] then
        stable = false
        table.insert(instability_reasons, "Not in interactive state (current: " .. tostring(G.STATE) .. ")")
    end
    if G.STATES ~= G.STATES.MENU and not G.STATE_COMPLETE then
        stable = false
        table.insert(instability_reasons, "State not complete (current: " .. tostring(G.STATE) .. ")")
    end
    
    -- Check for transition lock - prevents ready state during problematic transitions
    if BalatrobotAPI.checkTransitionLock() then
        stable = false
        table.insert(instability_reasons, "Transition lock active: " .. tostring(BalatrobotAPI.transition_lock.lock_reason))
    end
    
    -- Check for shop initialization - prevents ready state until shop is fully loaded on first entry
    if BalatrobotAPI.checkShopInitialization() then
        stable = false
        if BalatrobotAPI.shop_state.waiting_for_shop_init then
            table.insert(instability_reasons, "Waiting for shop initialization")
        end
        if BalatrobotAPI.shop_state.waiting_for_reroll then
            table.insert(instability_reasons, "Waiting for reroll completion")
        end
    end
    
    -- Check for pack state transition - prevents ready state until pack state is properly set
    if BalatrobotAPI.checkPackStateTransition() then
        stable = false
        table.insert(instability_reasons, "Waiting for pack state transition (999 -> " .. tostring(BalatrobotAPI.pack_state.target_pack_state) .. ")")
    end
    
    -- Check for pack cards readiness - prevents ready state until pack cards are fully loaded
    if not BalatrobotAPI.checkPackCardsReady() then
        stable = false
        table.insert(instability_reasons, "Waiting for pack cards to be ready")
    end
    
    -- Check for hand draw completion - prevents ready state until hand cards are drawn after tarot/spectral
    if BalatrobotAPI.checkHandDrawComplete() then
        stable = false
        table.insert(instability_reasons, "Waiting for hand cards to be drawn")
    end
    
    if not stable and #instability_reasons > 0 then
        sendDebugMessage("API not stable: " .. table.concat(instability_reasons, ", "))
    end
    
    return stable
end

function BalatrobotAPI.updatereadiness()
    -- Readiness requires both a stable game state AND the action execution flag to be false.
    local is_game_interactive = BalatrobotAPI.stablestate()
    local was_waiting = BalatrobotAPI.waitingForAction

    if is_game_interactive and not Actions.executing then
        if not BalatrobotAPI.waitingForAction then
            BalatrobotAPI.waitingForAction = true
            BalatrobotAPI.action_no = BalatrobotAPI.action_no + 1
            sendDebugMessage("API now ready for action (action_no: " .. BalatrobotAPI.action_no .. ")")
        end
    else
        if BalatrobotAPI.waitingForAction then
            BalatrobotAPI.waitingForAction = false
            local reasons = {}
            if not is_game_interactive then
                table.insert(reasons, "game not interactive")
            end
            if Actions.executing then
                table.insert(reasons, "action executing")
            end
            sendDebugMessage("API no longer ready: " .. table.concat(reasons, ", "))
        end
    end
end

-- Main update loop for the API, handles receiving and processing commands.
function BalatrobotAPI.update(dt)
    if not BalatrobotAPI.socket then
        BalatrobotAPI.socket = socket.udp()
        BalatrobotAPI.socket:settimeout(0)
        local port = arg[#arg] or BALATRO_BOT_CONFIG.port
        local port_num = tonumber(port)
        if not port_num then
            port_num = tonumber(BALATRO_BOT_CONFIG.port)
        end
        BalatrobotAPI.socket:setsockname('127.0.0.1', port_num)
    end

    data, msg_or_ip, port_or_nil = BalatrobotAPI.socket:receivefrom()
    if data then
        local command = data:match("^[^|]+")

        if command == 'STATUS' then
            local status = BalatrobotAPI.waitingForAction and "READY" or "BUSY"
            local debug_info = {
                status = status,
                game_state = tostring(G.STATE),
                state_complete = G.STATE_COMPLETE,
                controller_locked = (G.CONTROLLER and G.CONTROLLER.locked) or false,
                actions_executing = Actions.executing,
                transition_locked = BalatrobotAPI.transition_lock.locked,
                waiting_for_shop_init = BalatrobotAPI.shop_state.waiting_for_shop_init,
                waiting_for_reroll = BalatrobotAPI.shop_state.waiting_for_reroll,
                waiting_for_pack_state = BalatrobotAPI.pack_state.waiting_for_pack_state,
                target_pack_state = BalatrobotAPI.pack_state.target_pack_state,
                waiting_for_hand_draw = BalatrobotAPI.pack_state.waiting_for_hand_draw
            }
            sendDebugMessage("STATUS request - " .. status .. " (state: " .. tostring(G.STATE) .. ", actions_executing: " .. tostring(Actions.executing) .. ")")
            BalatrobotAPI.respond(debug_info)

        elseif command == 'GET_STATE' then
            if BalatrobotAPI.waitingForAction then
                BalatrobotAPI.notifyapiclient()
            else
                local reasons = {}
                if not BalatrobotAPI.stablestate() then
                    table.insert(reasons, "game state not stable")
                end
                if Actions.executing then
                    table.insert(reasons, "action currently executing")
                end
                sendDebugMessage("GET_STATE rejected - Bot is busy: " .. table.concat(reasons, ", "))
                BalatrobotAPI.respond({ error = "Bot is busy" })
            end
        else
            if not BalatrobotAPI.waitingForAction then
                local reasons = {}
                if not BalatrobotAPI.stablestate() then
                    table.insert(reasons, "game state not stable")
                end
                if Actions.executing then
                    table.insert(reasons, "action currently executing")
                end
                local error_msg = "Bot is not ready for actions: " .. table.concat(reasons, ", ")
                sendDebugMessage("Action rejected - " .. error_msg)
                BalatrobotAPI.respond({ error = error_msg })
                return
            end

            local _action = Utils.parseaction(data)
            local _err = Utils.validateAction(_action)
            if _err == Utils.ERROR.NOERROR then
                BalatrobotAPI.respond({response = "Action received: " .. _action[1]})
                sendDebugMessage("Set actions.executing to true for action: " .. _action[1])
                Actions.executing = true
                BalatrobotAPI.waitingForAction = false
                local dispatch_func = ACTION_DISPATCH[_action[1]]
                if dispatch_func then dispatch_func(_action) end
            else
                -- Handle validation errors - these are immediate failures
                local error_msg = ""
                if _err == Utils.ERROR.NUMPARAMS then
                    error_msg = "Error: Invalid number of parameters for action " .. _action[1]
                elseif _err == Utils.ERROR.MSGFORMAT then
                    error_msg = "Error: Invalid message format for action " .. _action[1]
                elseif _err == Utils.ERROR.INVALIDACTION then
                    error_msg = "Error: Invalid action " .. _action[1]
                end
                -- Set the action result for consistency
                Actions.last_action_success = false
                sendDebugMessage("Action validation failed: " .. error_msg)
                BalatrobotAPI.respond({ error = error_msg, action_result = false })
            end
        end
    end
end

function BalatrobotAPI.init()
    -- Add our new readiness updater to the main game update loop
    love.update = Hook.addcallback(love.update, BalatrobotAPI.updatereadiness)
    -- Add the main API update to the loop
    love.update = Hook.addcallback(love.update, BalatrobotAPI.update)

    -- Tell the game engine that every frame is 8/60 seconds long
    -- Speeds up the game execution
    -- Values higher than this seem to cause instability
    if BALATRO_BOT_CONFIG.dt then
        love.update = Hook.addbreakpoint(love.update, function(dt)
            return BALATRO_BOT_CONFIG.dt
        end)
    end

    -- Disable FPS cap
    if BALATRO_BOT_CONFIG.uncap_fps then
        G.FPS_CAP = 10000
    end

    -- Makes things move instantly instead of sliding
    if BALATRO_BOT_CONFIG.instant_move then
        function Moveable.move_xy(self, dt)
            -- Directly set the visible transform to the target transform
            self.VT.x = self.T.x
            self.VT.y = self.T.y
        end
    end

    -- Forcibly disable vsync
    if BALATRO_BOT_CONFIG.disable_vsync then
        love.window.setVSync(0)
    end

    -- Disable card scoring animation text
    if BALATRO_BOT_CONFIG.disable_card_eval_status_text then
        card_eval_status_text = function(card, eval_type, amt, percent, dir, extra) end
    end

    G.SETTINGS.GAMESPEED = 100
    G.SETTINGS.tutorial_complete = true
    
    -- Unlock all jokers and content
    G.PROFILES[G.SETTINGS.profile].all_unlocked = true
    for k, v in pairs(G.P_CENTERS) do
      if not v.demo and not v.wip then 
        v.alerted = true
        v.discovered = true
        v.unlocked = true
      end
    end
    for k, v in pairs(G.P_BLINDS) do
      if not v.demo and not v.wip then 
        v.alerted = true
        v.discovered = true
        v.unlocked = true
      end
    end
    for k, v in pairs(G.P_TAGS) do
      if not v.demo and not v.wip then 
        v.alerted = true
        v.discovered = true
        v.unlocked = true
      end
    end

    -- One-time event to set the initial state to complete
    G.E_MANAGER:add_event(Event({
        trigger = 'immediate', blocking = false,  blockable = false,
        func = function()
            if G.STATE == G.STATES.SPLASH then
                return false
            end
            G.STATE_COMPLETE = true
            return true
        end
    }))

    -- Only draw/present every Nth frame
    local original_draw = love.draw
    local draw_count = 0
    love.draw = function()
        draw_count = draw_count + 1
        if draw_count % BALATRO_BOT_CONFIG.frame_ratio == 0 then
            original_draw()
        end
    end

    local original_present = love.graphics.present
    love.graphics.present = function()
        if draw_count % BALATRO_BOT_CONFIG.frame_ratio == 0 then
            original_present()
        end
    end
end

return BalatrobotAPI