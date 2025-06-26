
Botlogger = { }
Botlogger.path = ''
Botlogger.filename = nil

function Botlogger.fileexists(filename)
    local _f = io.open(filename, 'r')
    if _f ~= nil then
        io.close(_f)
        return true
    else
        return false
    end
end

function Botlogger.getfilename(settings)
    local _filename = ''
    _filename = _filename .. tostring(settings.seed)
    _filename = _filename .. '_' .. tostring(settings.deck)
    _filename = _filename .. '_' .. tostring(settings.stake)
    _filename = _filename .. '_' .. tostring(settings.challenge)
    local port = arg[1] or BALATRO_BOT_CONFIG.port
    _filename = _filename .. '_' .. port
    _filename = _filename .. '.run'
    --_filename = Botlogger.path .. _filename
    sendDebugMessage(_filename)
    return _filename
end

function Botlogger.logbotdecision(...)
    local _args = {...}
    local _action = table.remove(_args, 1)
    local _logstring = ''
    
    if _action then
        for key,v in pairs(Bot.ACTIONS) do
            if _action == v then
                _logstring = key
                break
            end
        end

        for i = 1, #_args do
            _logstring = _logstring..'|'
            if type(_args[i]) == 'table' then
                for j = 1, #_args[i] do
                    _logstring = _logstring..','..tostring(_args[i][j])
                end
            else
                _logstring = _logstring..tostring(_args[i])
            end
        end

        if Botlogger.filename then
            local _f = io.open(Botlogger.filename, 'a')
            _f:write(_logstring, '\n')
            _f:close()
        end
    end
end

function Botlogger.start_run()
    if Bot.SETTINGS.replay and Bot.SETTINGS.replay == true then
        local _filename = Botlogger.getfilename(Bot.SETTINGS)
        if Botlogger.fileexists(_filename) then
            Botlogger.filename = _filename
        end
    else
        -- TODO if seed not specified, get it from the game
        local _filename = nil
        if Bot.SETTINGS.seed and Bot.SETTINGS.seed ~= '' then
            _filename = Botlogger.getfilename(Bot.SETTINGS)
        else
            local _settings = {
                stake = Bot.SETTINGS.stake,
                deck = Bot.SETTINGS.deck,
                seed = tostring(G.GAME.pseudorandom.seed),
                challenge = Bot.SETTINGS.challenge
            }
            _filename = Botlogger.getfilename(_settings)
        end

        if not Botlogger.fileexists(_filename) then
            Botlogger.filename = _filename
        end
    end
end

function Botlogger.init()

    
    
    

    -- Read replay file and populate action queues
    if Bot.SETTINGS.replay == true then
        local _replayfile = Botlogger.getfilename(Bot.SETTINGS)

        if Botlogger.fileexists(_replayfile) then

            local _num_action = 0
            for line in io.lines(_replayfile) do
                _num_action = _num_action + 1
                
                local _action = Utils.parseaction(line)
                sendDebugMessage(line)
                local _params = Bot.ACTIONPARAMS[_action[1]]
                for i = 2, #_action do
                    sendDebugMessage(tostring(_action[i][1]))
                end
                List.pushleft(Botlogger['q_'.._params.func], { _num_action, _action })
            end           
        end
    elseif Bot.SETTINGS.replay == false then
        
    end

    -- TODO Hook run start/end
    G.FUNCS.start_run = Hook.addcallback(G.FUNCS.start_run, Botlogger.start_run)
end

return Botlogger