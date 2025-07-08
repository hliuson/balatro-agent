
local socket = require "socket"

local data, msg_or_ip, port_or_nil

BalatrobotAPI = { }
BalatrobotAPI.socket = nil

-- waitingForAction: true when the mod is in a stable, interactive state ready for a command
-- waitingForAction: false when it is busy processing a command or waiting for animations
BalatrobotAPI.waitingForAction = false -- Initially, we are ready for an action: start the game
BalatrobotAPI.action_no = 0

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
    [G.STATES.NEW_ROUND] = true,
}


-- Dispatch table to call the appropriate function in actions.lua based on the action enum
local ACTION_DISPATCH = {
    [Bot.ACTIONS.SELECT_BLIND] = function(action) Actions.select_blind() end,
    [Bot.ACTIONS.SKIP_BLIND] = function(action) Actions.skip_blind() end,
    [Bot.ACTIONS.PLAY_HAND] = function(action) Actions.play_hand(action[2]) end,
    [Bot.ACTIONS.DISCARD_HAND] = function(action) Actions.discard_hand(action[2]) end,
    [Bot.ACTIONS.END_SHOP] = function(action) Actions.end_shop() end,
    [Bot.ACTIONS.REROLL_SHOP] = function(action) Actions.reroll_shop() end,
    [Bot.ACTIONS.BUY_CARD] = function(action) Actions.buy_card(action[2][1]) end,
    [Bot.ACTIONS.BUY_VOUCHER] = function(action) Actions.buy_voucher(action[2][1]) end,
    [Bot.ACTIONS.BUY_BOOSTER] = function(action) Actions.buy_booster(action[2][1]) end,
    [Bot.ACTIONS.SELECT_BOOSTER_CARD] = function(action) Actions.select_booster_card(action[2][1], action[3]) end,
    [Bot.ACTIONS.SKIP_BOOSTER_PACK] = function(action) Actions.skip_booster_pack() end,
    [Bot.ACTIONS.SELL_JOKER] = function(action) Actions.sell_joker(action[2]) end,
    [Bot.ACTIONS.USE_CONSUMABLE] = function(action) Actions.use_consumable(action[2], action[3]) end,
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

function BalatrobotAPI.stablestate()
    -- Check if the game state is stable and not in a transition
    local stable = true
    if not G.CONTROLLER or G.CONTROLLER.locked then
        stable = false
    end
    if not INTERACTIVE_STATES[G.STATE] then
        stable = false
    end
    if G.STATES ~= G.STATES.MENU and not G.STATE_COMPLETE then
        stable = false
    end
    return stable
end

function BalatrobotAPI.updatereadiness()
    -- Readiness requires both a stable game state AND the action execution flag to be false.
    local is_game_interactive = BalatrobotAPI.stablestate()

    if is_game_interactive and not Actions.executing then
        if not BalatrobotAPI.waitingForAction then
            BalatrobotAPI.waitingForAction = true
            BalatrobotAPI.action_no = BalatrobotAPI.action_no + 1
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
            BalatrobotAPI.respond({ status = status})

        elseif command == 'GET_STATE' then
            if BalatrobotAPI.waitingForAction then
                BalatrobotAPI.notifyapiclient()
            else
                BalatrobotAPI.respond({ error = "Bot is busy" })
            end
        else
            if not BalatrobotAPI.waitingForAction then
                BalatrobotAPI.respond({ error = "Bot is not ready for actions" })
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
                -- Handle validation errors
                if _err == Utils.ERROR.NUMPARAMS then
                    BalatrobotAPI.respond({ error = "Error: Invalid number of parameters for action " .. _action[1] })
                elseif _err == Utils.ERROR.MSGFORMAT then
                    BalatrobotAPI.respond({ error = "Error: Invalid message format for action " .. _action[1] })
                elseif _err == Utils.ERROR.INVALIDACTION then
                    BalatrobotAPI.respond({ error = "Error: Invalid action " .. _action[1] })
                end
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
        G.FPS_CAP = 1000
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