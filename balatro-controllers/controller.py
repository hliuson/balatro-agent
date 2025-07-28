import sys
import json
import socket
import time
import os
import platform
from enum import Enum
#from gamestates import cache_state
import subprocess
import random
import threading
import socket
import atexit
import weakref
import numpy as np

_port_lock = threading.Lock()
_used_ports = set()

# Track active controllers for cleanup on exit
_active_controllers = weakref.WeakSet()

def _cleanup_all_controllers():
    """Emergency cleanup function called on process exit"""
    for controller in list(_active_controllers):
        try:
            controller.close()
        except:
            pass  # Ignore errors during emergency cleanup

# Register cleanup for process exit
atexit.register(_cleanup_all_controllers)

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except socket.error:
            return False
        
def get_available_port():
    """Thread-safe port allocation for parallel execution."""
    with _port_lock:
        for port in range(12346, 65536):
            if port not in _used_ports and is_port_available(port):
                _used_ports.add(port)
                return port
        raise RuntimeError("No available ports")

def release_port(port):
    """Release a port when controller is closed."""
    with _port_lock:
        _used_ports.discard(port)

def format_card(card):
    """Format a single card for display."""
    if not card:
        return "None"
    
    # Check if card is face down - if so, hide all info except basic status
    if card.get('facing') == 'back':
        name = "Face Down Card"
        # Add selection status for face down cards
        if card.get('highlighted'):
            name += " [SELECTED]"
        return name
    
    # Check if card is debuffed
    debuffed = card.get('debuff', False)
    
    value = card.get('value', '')
    #map ace/face cards to self, map numbers i.e. '2' to "Two"
    value_map = {
        'Ace': 'Ace', '2': 'Two', '3': 'Three', '4': 'Four',
        '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight',
        '9': 'Nine', '10': 'Ten', 'Jack': 'Jack', 'Queen': 'Queen',
        'King': 'King'
    }
    value = value_map.get(value, value)  # Use the mapped value or original if not found
    suit = card.get('suit', '')
    name = f"{value} of {suit}" if value and suit else card.get('name', 'Unknown Card')
    
    # Add debuff status early if present
    if debuffed:
        name += " [DEBUFFED]"
    
    # Add enhancement
    enhancement = card.get('ability_name', '')
    if enhancement and enhancement != "Default Base":
        name += f" ({enhancement})"
    if enhancement == "Stone Card":
        name = "Stone Card"

    # Add seal
    seal = card.get('seal', 'none')
    if seal and seal != "none":
        name += f" [{seal} Seal]"
        
    # Add edition
    edition = card.get('edition', {})
    if edition:
        if edition.get('foil'):
            name += " [Foil]"
        if edition.get('holo'):
            name += " [Holographic]"
        if edition.get('polychrome'):
            name += " [Polychrome]"
        if edition.get('negative'):
            name += " [Negative]"
            
    # Add stickers (eternal, perishable, rental)
    if card.get('eternal'):
        name += " [Eternal]"
    if card.get('perishable'):
        name += f" [Perishable {card.get('perish_tally', '?')}]"
    if card.get('rental'):
        name += " [Rental]"
        
    # Add selection status for hand cards
    if card.get('highlighted'):
        name += " [SELECTED]"
        
    # Add description text if available
    description = card.get('description_text', '')
    if description:
        name += f" - {description}"

    return name

def format_cards(cards):
    """Format a list of cards for display."""
    if not cards:
        return "None"
    cards_formatted = [format_card(card) for card in cards]
    # 1-indexed list, one card per line
    return "\n".join([f"{i+1}. {card}" for i, card in enumerate(cards_formatted)])


def format_jokers(jokers):
    """Format jokers and return list of strings."""
    if not jokers:
        return []
    formatted_jokers = []
    for joker in jokers:
        name = joker.get('name', 'Unknown Joker')
        sell_value = joker.get('cost', 0)
        description = joker.get('description_text', '')
        
        # Add edition
        edition = joker.get('edition', {})
        if edition:
            if edition.get('foil'):
                name += " [Foil]"
            if edition.get('holo'):
                name += " [Holographic]"
            if edition.get('polychrome'):
                name += " [Polychrome]"
            if edition.get('negative'):
                name += " [Negative]"
                
        # Add stickers (eternal, perishable, rental)
        if joker.get('eternal'):
            name += " [Eternal]"
        if joker.get('perishable'):
            name += f" [Perishable {joker.get('perish_tally', '?')}]"
        if joker.get('rental'):
            name += " [Rental]"
        
        if description:
            formatted_jokers.append(f"{name} (Sell Value: {sell_value}) - {description}")
        else:
            formatted_jokers.append(f"{name} (Sell Value: {sell_value})")
    
    return formatted_jokers

def format_shop_cards(shop_cards):
    """Format shop cards (jokers, playing cards, or consumables) and return list of strings."""
    if not shop_cards:
        return []
    formatted_cards = []
    for card in shop_cards:
        name = card.get('name', 'Unknown Card')
        cost = card.get('cost', 0)
        description = card.get('description_text', '')
        card_set = card.get('set', '')
        
        # Handle different card types
        if card_set == 'Joker':
            # Joker formatting - include sell value if available
            sell_value = card.get('sell_cost', 0)
            if sell_value > 0:
                cost_info = f" (Cost: {cost}, Sell: {sell_value})"
            else:
                cost_info = f" (Cost: {cost})"
        elif card_set in ['Tarot', 'Spectral', 'Planet']:
            # Consumable formatting - add type prefix
            name = f"[{card_set}] {name}"
            cost_info = f" (Cost: {cost})"
        else:
            # Playing cards or other types
            value = card.get('value', '')
            suit = card.get('suit', '')
            if value and suit:
                # Format playing card name
                value_map = {
                    'Ace': 'Ace', '2': 'Two', '3': 'Three', '4': 'Four',
                    '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight',
                    '9': 'Nine', '10': 'Ten', 'Jack': 'Jack', 'Queen': 'Queen',
                    'King': 'King'
                }
                value = value_map.get(value, value)
                name = f"{value} of {suit}"
            
            # Add enhancement for playing cards
            enhancement = card.get('ability_name', '')
            if enhancement and enhancement != "Default Base":
                name += f" ({enhancement})"
            if enhancement == "Stone Card":
                name = "Stone Card"
                
            # Add seal for playing cards
            seal = card.get('seal', 'none')
            if seal and seal != "none":
                name += f" [{seal} Seal]"
            
            cost_info = f" (Cost: {cost})"
        
        # Add edition (applies to all card types)
        edition = card.get('edition', {})
        if edition:
            if edition.get('foil'):
                name += " [Foil]"
            if edition.get('holo'):
                name += " [Holographic]"
            if edition.get('polychrome'):
                name += " [Polychrome]"
            if edition.get('negative'):
                name += " [Negative]"
                
        # Add stickers (applies to all card types)
        if card.get('eternal'):
            name += " [Eternal]"
        if card.get('perishable'):
            name += f" [Perishable {card.get('perish_tally', '?')}]"
        if card.get('rental'):
            name += " [Rental]"
        
        if description:
            formatted_cards.append(f"{name}{cost_info} - {description}")
        else:
            formatted_cards.append(f"{name}{cost_info}")
    
    return formatted_cards

def format_boosters(boosters):
    """Format booster packs and return list of strings."""
    if not boosters:
        return []
    formatted_boosters = []
    for booster in boosters:
        name = booster.get('name', 'Unknown Booster')
        cost = booster.get('cost', 0)
        kind = booster.get('kind', '')
        
        # Add pack type info if available
        extra = booster.get('extra', 0)
        choose = booster.get('choose', 0)
        if extra and choose:
            pack_info = f" ({extra} cards, choose {choose})"
        elif extra:
            pack_info = f" ({extra} cards)"
        else:
            pack_info = ""
        
        # Add edition
        edition = booster.get('edition', {})
        if edition:
            if edition.get('foil'):
                name += " [Foil]"
            if edition.get('holo'):
                name += " [Holographic]"
            if edition.get('polychrome'):
                name += " [Polychrome]"
            if edition.get('negative'):
                name += " [Negative]"
        
        formatted_boosters.append(f"{name}{pack_info} (Cost: {cost})")
    
    return formatted_boosters

def format_vouchers(vouchers):
    """Format vouchers and return list of strings."""
    if not vouchers:
        return []
    formatted_vouchers = []
    for voucher in vouchers:
        name = voucher.get('name', 'Unknown Voucher')
        cost = voucher.get('cost', 0)
        description = voucher.get('description_text', '')
        
        # Add edition
        edition = voucher.get('edition', {})
        if edition:
            if edition.get('foil'):
                name += " [Foil]"
            if edition.get('holo'):
                name += " [Holographic]"
            if edition.get('polychrome'):
                name += " [Polychrome]"
            if edition.get('negative'):
                name += " [Negative]"
        
        if description:
            formatted_vouchers.append(f"{name} (Cost: {cost}) - {description}")
        else:
            formatted_vouchers.append(f"{name} (Cost: {cost})")
    
    return formatted_vouchers

def format_consumables(consumables):
    """Format consumable cards (tarot, spectral, planet) and return list of strings."""
    if not consumables:
        return []
    formatted_consumables = []
    for consumable in consumables:
        name = consumable.get('name', 'Unknown Consumable')
        card_set = consumable.get('set', '')
        description = consumable.get('description_text', '')
        
        # Add card type prefix for clarity
        if card_set in ['Tarot', 'Spectral', 'Planet']:
            name = f"[{card_set}] {name}"
        
        # Add edition
        edition = consumable.get('edition', {})
        if edition:
            if edition.get('foil'):
                name += " [Foil]"
            if edition.get('holo'):
                name += " [Holographic]"
            if edition.get('polychrome'):
                name += " [Polychrome]"
            if edition.get('negative'):
                name += " [Negative]"
        
        # Add stickers (eternal, perishable, rental)
        if consumable.get('eternal'):
            name += " [Eternal]"
        if consumable.get('perishable'):
            name += f" [Perishable {consumable.get('perish_tally', '?')}]"
        if consumable.get('rental'):
            name += " [Rental]"

        if description:
            formatted_consumables.append(f"{name} - {description}")
        else:
            formatted_consumables.append(name)
    
    return formatted_consumables

def format_pack_cards(pack_cards):
    """Format pack cards (tarot, spectral, planet, joker, playing cards) for display."""
    if not pack_cards:
        return "None"
    formatted_cards = []
    for i, card in enumerate(pack_cards):
        name = card.get('name', 'Unknown Card')
        description = card.get('description_text', '')
        card_set = card.get('set', '')
        
        # Handle different card types
        if card_set in ['Tarot', 'Spectral', 'Planet']:
            # Consumable cards - just name and description
            pass
        elif card_set == 'Joker':
            # Apply joker formatting
            sell_value = card.get('sell_cost', 0)
            # Add edition
            edition = card.get('edition', {})
            if edition:
                if edition.get('foil'):
                    name += " [Foil]"
                if edition.get('holo'):
                    name += " [Holographic]"
                if edition.get('polychrome'):
                    name += " [Polychrome]"
                if edition.get('negative'):
                    name += " [Negative]"
                    
            # Add stickers (eternal, perishable, rental)
            if card.get('eternal'):
                name += " [Eternal]"
            if card.get('perishable'):
                name += f" [Perishable {card.get('perish_tally', '?')}]"
            if card.get('rental'):
                name += " [Rental]"
        else:
            # Playing cards - apply enhancement, seal, edition formatting
            value = card.get('value', '')
            suit = card.get('suit', '')
            if value and suit:
                value_map = {
                    'Ace': 'Ace', '2': 'Two', '3': 'Three', '4': 'Four',
                    '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight',
                    '9': 'Nine', '10': 'Ten', 'Jack': 'Jack', 'Queen': 'Queen',
                    'King': 'King'
                }
                value = value_map.get(value, value)
                name = f"{value} of {suit}"
            
            # Add enhancement
            enhancement = card.get('ability_name', '')
            if enhancement and enhancement != "Default Base":
                name += f" ({enhancement})"
            if enhancement == "Stone Card":
                name = "Stone Card"
                
            # Add seal
            seal = card.get('seal', 'none')
            if seal and seal != "none":
                name += f" [{seal} Seal]"
            
            # Add edition
            edition = card.get('edition', {})
            if edition:
                if edition.get('foil'):
                    name += " [Foil]"
                if edition.get('holo'):
                    name += " [Holographic]"
                if edition.get('polychrome'):
                    name += " [Polychrome]"
                if edition.get('negative'):
                    name += " [Negative]"
                    
            # Add stickers (eternal, perishable, rental)
            if card.get('eternal'):
                name += " [Eternal]"
            if card.get('perishable'):
                name += f" [Perishable {card.get('perish_tally', '?')}]"
            if card.get('rental'):
                name += " [Rental]"
        
        if description:
            formatted_cards.append(f"{i+1}. {name} - {description}")
        else:
            formatted_cards.append(f"{i+1}. {name}")
    
    return "\n".join(formatted_cards)

def format_game_state(state) -> str:
    """Format the game state for display, omitting zero or irrelevant fields.
    The game state includes:
     - Game Info: Round, Ante, Money, Chips, Boss (always)
     - Round Info: Hands Left, Discards Left (if in SELECTING_HAND state)
     - Hand: List of cards in hand (if in SELECTING_HAND, TAROT_PACK, or SPECTRAL_PACK state)
     - Deck: List of cards in deck (always)
     - Shop: List of cards in shop, Vouchers, Boosters (if in SHOP state)
     - booster pack contents (if in any PACK state)
    """
    if not isinstance(state, dict):
        raise ValueError("State must be a dictionary, got: {}".format(type(state)))

    output = []

    output.append("== Game State ==")
    #json.dump(state, sys.stdout, indent=4)  # Print the raw state for debugging
    game_state_enum = State(state['state'])
    if game_state_enum == State.SELECTING_HAND:
        output.append("Current State: SELECTING_HAND. Play or discard cards.")
    elif game_state_enum == State.SHOP:
        output.append("Current State: SHOP. Buy cards, vouchers, or boosters. When done, use END_SHOP or REROLL_SHOP.")
    else:
        output.append(f"Current State: {game_state_enum.name}")
    if state.get("game"):
        game = state["game"]
        output.append("== Game Info ==")
        output.append(f"Round: {game.get('round', 0)}, Ante: {game.get('ante', 0)}")
        output.append(f"Money: ${game.get('dollars', 0)}")
        output.append(f"Chips: {game.get('chips', 0)}")

    if state.get("round"):
        round_info = state["round"]
        output.append("\n== Round Info ==")
        output.append(f"Chips: {round_info.get('chips', 0)}")
        output.append(f"Hands Left: {round_info.get('hands_left', 0)}, Discards Left: {round_info.get('discards_left', 0)}")
        
        # Display current selected hand information if available
        if round_info.get("current_hand") and round_info["current_hand"].get("handname"):
            current_hand = round_info["current_hand"]
            hand_name = current_hand.get("handname", "")
            hand_chips = current_hand.get("chips", 0)
            hand_mult = current_hand.get("mult", 0)
            hand_level = current_hand.get("level", 0)
            
            if hand_name:  # Only display if we have a valid hand name
                output.append(f"Selected Hand: {hand_name} (Level {hand_level}) - {hand_chips} chips, {hand_mult} mult")
    if state.get("ante"):
        ante = state["ante"]
        blinds = ante["blinds"]
        currentblind = blinds["current"]
        chips = currentblind["chips"]
        output.append(f"Chips to pass this round: {chips}")

    if state.get("hand"):
        output.append("\n== Your Hand ==")
        output.append(format_cards(state["hand"]))

    if state.get("jokers"):
        output.append("\n== Jokers ==")
        jokers_list = format_jokers(state["jokers"])
        if jokers_list:
            output.append(", ".join(jokers_list))
        else:
            output.append("None")

    if state.get("consumables"):
        output.append("\n== Consumables ==")
        consumables_list = format_consumables(state["consumables"])
        if consumables_list:
            output.append(", ".join(consumables_list))
        else:
            output.append("None")

    if state.get("pack_cards"):
        output.append("\n== Pack Cards ==")
        output.append(format_pack_cards(state["pack_cards"]))

    if state.get("shop") and state["state"] == State.SHOP.value:
        shop = state["shop"]
        output.append("\n== Shop ==")
        
        # G.shop_jokers contains mixed card types (jokers, consumables, playing cards)
        if shop.get("jokers"):
            shop_cards_list = format_shop_cards(shop['jokers'])
            if shop_cards_list:
                output.append(f"Shop Cards: {', '.join(shop_cards_list)}")
            else:
                output.append("Shop Cards: None")
        
        if shop.get("vouchers"):
            vouchers_list = format_vouchers(shop['vouchers'])
            if vouchers_list:
                output.append(f"Vouchers: {', '.join(vouchers_list)}")
            else:
                output.append("Vouchers: None")
        
        if shop.get("boosters"):
            boosters_list = format_boosters(shop['boosters'])
            if boosters_list:
                output.append(f"Boosters: {', '.join(boosters_list)}")
            else:
                output.append("Boosters: None")
    
    return "\n".join(output)

class State(Enum): # these enums are lifted from the game code so DO NOT CHANGE THEM
    SELECTING_HAND = 1
    HAND_PLAYED = 2
    DRAW_TO_HAND = 3
    GAME_OVER = 4
    SHOP = 5
    PLAY_TAROT = 6
    BLIND_SELECT = 7
    ROUND_EVAL = 8
    TAROT_PACK = 9
    PLANET_PACK = 10
    MENU = 11
    TUTORIAL = 12
    SPLASH = 13
    SANDBOX = 14
    SPECTRAL_PACK = 15
    DEMO_CTA = 16
    STANDARD_PACK = 17
    BUFFOON_PACK = 18
    NEW_ROUND = 19
    UNKNOWN = 999

EXPECTED_STATE_COMPONENTS = {
    State.SELECTING_HAND: ["hand", "round"],
    State.SHOP: ["shop"],
    State.TAROT_PACK: ["hand", "pack_cards"],
    State.SPECTRAL_PACK: ["hand", "pack_cards"],
    State.PLANET_PACK: ["pack_cards"],
    State.STANDARD_PACK: ["pack_cards"],
    State.BUFFOON_PACK: ["pack_cards"],
}

class Actions(Enum): # these enums are from the lua mod. you can add new ones but it requires changes in the lua mod as well
    SELECT_BLIND = 1
    SKIP_BLIND = 2
    SELECT_HAND_CARD = 3
    CLEAR_HAND_SELECTION = 4
    PLAY_SELECTED = 5
    DISCARD_SELECTED = 6
    END_SHOP = 7
    REROLL_SHOP = 8
    BUY_CARD = 9
    BUY_VOUCHER = 10
    BUY_BOOSTER = 11
    SELECT_BOOSTER_CARD = 12
    SKIP_BOOSTER_PACK = 13
    SELL_JOKER = 14
    USE_CONSUMABLE = 15
    SELL_CONSUMABLE = 16
    REARRANGE_JOKERS = 17
    REARRANGE_CONSUMABLES = 18
    REARRANGE_HAND = 19
    PASS = 20
    START_RUN = 21
    RETURN_TO_MENU = 22
    CASH_OUT = 23


class BalatroControllerBase:
    def __init__(
        self,
        verbose = False,
        policy_states = [],
        auto_start = True,
        silence_lovely = True
    ):
        self.G = None
        self.port = get_available_port()
        self.addr = ("localhost", self.port)
        self.balatro_instance = None
        self.sock = None
        self.verbose = verbose
        self.policy_states = policy_states
        self.first_run = True
        self.silence_lovely = silence_lovely

        # State handlers are now expected to be populated by subclasses
        self.state_handlers = {}
        #init state handlers for non-interactive states
        self.state_handlers[State.HAND_PLAYED] = self.pass_action
        self.state_handlers[State.DRAW_TO_HAND] = self.pass_action
        self.state_handlers[State.NEW_ROUND] = self.pass_action
        self.connected = False
        
        self.display = None
        # Register for cleanup
        _active_controllers.add(self)
        
        if auto_start:
            self.start_balatro_instance()

    def pass_action(self, state):
        """
        A no-op action that can be used to pass the turn or skip an action.
        This is useful in cases where the bot does not want to take any action.
        """
        return [Actions.PASS]

    def get_status(self):
        if not self.connected:
            raise ConnectionError("Not connected to Balatro instance.")
        self.sock.sendto(bytes("STATUS", "utf-8"), self.addr)
        starttime = time.time()
        wait_sec = 0.5
        data = None
        while starttime + wait_sec > time.time():
            try:
                data, _ = self.sock.recvfrom(65536)
                break
            except socket.timeout:
                time.sleep(0.01)
        if not data:
            raise ConnectionError("No response from Balatro instance.")
        data = json.loads(data)
        response = data.get('response')
        if response:
            status = response.get('status')
            error = response.get('error')
            if status:
                return status
            if error:
                raise ConnectionError(f"Error from Balatro instance: {error}")
        
        raise ConnectionError("Invalid response from Balatro instance. Expected 'status' field.")

    def get_state(self):
        if not self.connected:
            raise ConnectionError("Not connected to Balatro instance.")

        self.sock.sendto(bytes("GET_STATE", "utf-8"), self.addr)
        data, _ = self.sock.recvfrom(65536)
        return json.loads(data)

    def get_next_display(self):
        """Find next available display number starting from 99 - thread-safe"""
        with _port_lock:  # Reuse the same lock for simplicity
            for display_num in range(99, 0, -1):
                if display_num in getattr(self.__class__, '_used_displays', set()):
                    continue
                result = subprocess.run(['pgrep', '-f', f'Xvfb :{display_num}'], 
                                      capture_output=True)
                if result.returncode != 0:  # Display not in use
                    # Track used displays at class level
                    if not hasattr(self.__class__, '_used_displays'):
                        self.__class__._used_displays = set()
                    self.__class__._used_displays.add(display_num)
                    return display_num
            raise RuntimeError("No available displays")

    def start_virtual_display(self, display_num):
        """Start Xvfb on specific display"""
        self.display_process = subprocess.Popen(
            ['Xvfb', f':{display_num}', '-screen', '0', '1024x768x24'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(0.5)
        return f':{display_num}'

    def cleanup_display(self, display_num):
        """Clean up virtual display"""
        if hasattr(self, 'display_process') and self.display_process:
            self.display_process.terminate()
            self.display_process.wait()
        # Remove from used displays
        if hasattr(self.__class__, '_used_displays'):
            self.__class__._used_displays.discard(display_num)

    def setup_display_for_linux(self):
        """Set up display for Linux - always use virtual display for consistency"""
        if platform.system() != "Linux":
            return
            
        # Always use virtual display for headless operation
        self.display_num = self.get_next_display()
        display = self.start_virtual_display(self.display_num)
        
        # Set DISPLAY in our environment
        if not hasattr(self, 'balatro_env') or self.balatro_env is None:
            self.balatro_env = os.environ.copy()
        self.balatro_env['DISPLAY'] = display
        self.display = display
        if self.verbose:
            print(f"Using virtual display: {display}")

    def start_balatro_instance(self):
        print("Starting Balatro instance...") if self.verbose else None
        
        if platform.system() == "Linux":
            if not os.getenv('XDG_RUNTIME_DIR'):
                os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'
                os.makedirs('/tmp/runtime-root', exist_ok=True)

        # Get Balatro executable path from environment variable
        balatro_exec_path = os.getenv('BALATRO_EXEC_PATH')
        
        if not balatro_exec_path:
            # Fallback to default paths based on platform
            if platform.system() == "Windows":
                balatro_exec_path = r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
            elif platform.system() == "Linux":
                # Check if we're in the Docker container with Love2D setup
                if os.path.exists("/app/balatro/bin/love") and os.path.exists("/app/balatro/bin/Balatro.love"):
                    balatro_exec_path = "/app/balatro/bin/Balatro.love"
                else:
                    # Fallback to Steam installation
                    balatro_exec_path = os.path.expanduser("~/.steam/steam/steamapps/common/Balatro/Balatro")
            elif platform.system() == "Darwin":  # macOS
                balatro_exec_path = os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/MacOS/Balatro")
            else:
                raise Exception(f"Unsupported platform: {platform.system()}")
        print(f"Using Balatro executable at: {balatro_exec_path}") if self.verbose else None
        # Check if the executable exists
        if not os.path.exists(balatro_exec_path):
            raise FileNotFoundError(f"Balatro executable not found at: {balatro_exec_path}")
        
        # Build the command
        if platform.system() == "Linux" and balatro_exec_path == "/app/balatro/bin/Balatro.love":
            print("Assuming Linux environment for Balatro execution...") if self.verbose else None
            # Use Love2D directly with lovely preload for mod support
            cmd = ["./bin/love", "./bin/Balatro.love", str(self.port)]
            # Set working directory and environment for lovely
            self.balatro_working_dir = "/app/balatro"
            self.balatro_env = os.environ.copy()
            self.balatro_env["LD_PRELOAD"] = "./liblovely.so"
        else:
            # Standard format: executable port
            cmd = [balatro_exec_path, str(self.port)]
            self.balatro_working_dir = None
            self.balatro_env = None
        
        # Set up display for Linux (always use virtual display)
        self.setup_display_for_linux()
        
        if self.verbose:
            print(f"Starting Balatro with command: {' '.join(cmd)}")
        
        stdout = None if not self.silence_lovely else subprocess.DEVNULL
        stderr = None if not self.silence_lovely else subprocess.DEVNULL
        
        self.balatro_instance = subprocess.Popen(
            cmd, 
            cwd=self.balatro_working_dir, 
            env=self.balatro_env,
            stdout=stdout,
            stderr=stderr
        )
        if self.verbose:
            print(f"Balatro instance started with PID: {self.balatro_instance.pid} on port {self.port}")

        self.pid = self.balatro_instance.pid

        max_attempts = 60
        attempt = 0
        while attempt < max_attempts:
            try:
                if self.verbose:
                    print(f"Attempting to connect to Balatro instance (attempt {attempt + 1}/{max_attempts})...")
                self.connect_socket()
                if self.verbose:
                    print("Successfully connected to Balatro instance.")
                return
            except Exception as e:
                self.connected = False # Ensure connected is False on failure
                if self.verbose:
                    print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                time.sleep(0.1)
            attempt += 1
        
        raise Exception("Failed to connect to Balatro instance after multiple attempts")

    def stop_balatro_instance(self):
        if self.balatro_instance:
            self.balatro_instance.kill()

    def sendcmd(self, cmd, **kwargs):
        if not self.connected:
            raise ConnectionError("Not connected to Balatro instance.")
        if self.verbose:
            print(f"Sending command: {cmd}")
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def actionToCmd(self, action):
        result = []

        for x in action:
            if isinstance(x, Actions):
                result.append(x.name)
            elif type(x) is list:
                result.append(",".join([str(y) for y in x]))
            else:
                result.append(str(x))

        return "|".join(result)

    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))

    def get_valid_actions(self, game_state):
        valid_actions = []
        current_state = State(game_state['state'])

        if current_state == State.SELECTING_HAND:
            # Hand selection actions
            valid_actions.append({
                "action": Actions.SELECT_HAND_CARD.name,
                "params": [
                    {
                        "name": "card_index",
                        "type": "int",
                        "required": True,
                        "constraints": {
                            "min_value": 1,
                            "max_value": len(game_state.get('hand', [])),
                            "card_source": "hand"
                        }
                    }
                ]
            })
            valid_actions.append({
                "action": Actions.CLEAR_HAND_SELECTION.name,
                "params": []
            })
            valid_actions.append({
                "action": Actions.PLAY_SELECTED.name,
                "params": []
            })
            valid_actions.append({
                "action": Actions.DISCARD_SELECTED.name,
                "params": []
            })
        elif current_state == State.SHOP:
            # END_SHOP action
            valid_actions.append({
                "action": Actions.END_SHOP.name,
                "params": []
            })
            
            # REROLL_SHOP action
            valid_actions.append({
                "action": Actions.REROLL_SHOP.name,
                "params": []
            })
            
            # BUY_CARD action (for buying jokers from shop)
            shop_jokers = game_state.get('shop', {}).get('jokers', [])
            if shop_jokers:
                valid_actions.append({
                    "action": Actions.BUY_CARD.name,
                    "params": [
                        {
                            "name": "card_index",
                            "type": "int",
                            "required": True,
                            "constraints": {
                                "min_value": 1,
                                "max_value": len(shop_jokers),
                                "card_source": "shop_jokers"
                            }
                        }
                    ]
                })
            
            # BUY_VOUCHER action
            shop_vouchers = game_state.get('shop', {}).get('vouchers', [])
            if shop_vouchers:
                valid_actions.append({
                    "action": Actions.BUY_VOUCHER.name,
                    "params": [
                        {
                            "name": "voucher_index",
                            "type": "int",
                            "required": True,
                            "constraints": {
                                "min_value": 1,
                                "max_value": len(shop_vouchers),
                                "card_source": "shop_vouchers"
                            }
                        }
                    ]
                })
            
            # BUY_BOOSTER action
            shop_boosters = game_state.get('shop', {}).get('boosters', [])
            if shop_boosters:
                valid_actions.append({
                    "action": Actions.BUY_BOOSTER.name,
                    "params": [
                        {
                            "name": "booster_index",
                            "type": "int",
                            "required": True,
                            "constraints": {
                                "min_value": 1,
                                "max_value": len(shop_boosters),
                                "card_source": "shop_boosters"
                            }
                        }
                    ]
                })
            
            # SELL_JOKER action
            player_jokers = game_state.get('jokers', [])
            if player_jokers:
                valid_actions.append({
                    "action": Actions.SELL_JOKER.name,
                    "params": [
                        {
                            "name": "joker_index",
                            "type": "int",
                            "required": True,
                            "constraints": {
                                "min_value": 1,
                                "max_value": len(player_jokers),
                                "card_source": "jokers"
                            }
                        }
                    ]
                })
            
            # SELL_CONSUMABLE action
            player_consumables = game_state.get('consumables', [])
            if player_consumables:
                valid_actions.append({
                    "action": Actions.SELL_CONSUMABLE.name,
                    "params": [
                        {
                            "name": "consumable_index",
                            "type": "int",
                            "required": True,
                            "constraints": {
                                "min_value": 1,
                                "max_value": len(player_consumables),
                                "card_source": "consumables"
                            }
                        }
                    ]
                })
        elif current_state == State.MENU:
            valid_actions.append({
                "action": Actions.START_RUN.name,
                "params": [
                    {"name": "stake", "type": "int", "required": False, "default": 1},
                    {"name": "deck", "type": "str", "required": False, "default": "Red Deck"},
                    {"name": "seed", "type": "str", "required": False, "default": None},
                    {"name": "challenge", "type": "str", "required": False, "default": None}
                ]
            })
        elif current_state == State.BLIND_SELECT:
            valid_actions.append({
                "action": Actions.SELECT_BLIND.name,
                "params": []
            })
            valid_actions.append({
                "action": Actions.SKIP_BLIND.name,
                "params": []
            })
        elif current_state == State.GAME_OVER:
            valid_actions.append({
                "action": Actions.RETURN_TO_MENU.name,
                "params": []
            })
            valid_actions.append({
                "action": Actions.START_RUN.name,
                "params": [
                    {"name": "stake", "type": "int", "required": False, "default": 1},
                    {"name": "deck", "type": "str", "required": False, "default": "Red Deck"},
                    {"name": "seed", "type": "str", "required": False, "default": None},
                    {"name": "challenge", "type": "str", "required": False, "default": None}
                ]
            })
        elif current_state == State.ROUND_EVAL:
            valid_actions.append({
                "action": Actions.CASH_OUT.name,
                "params": []
            })
        elif current_state in [State.TAROT_PACK, State.SPECTRAL_PACK]:
            # Select hand card to target
            valid_actions.append({
                "action": Actions.SELECT_HAND_CARD.name,
                "params": [
                    {
                        "name": "card_index",
                        "type": "int",
                        "required": True,
                        "constraints": {
                            "min_value": 1,
                            "max_value": len(game_state.get('hand', [])),
                            "card_source": "hand"
                        }
                    }
                ]
            })
            # Select booster card to use
            valid_actions.append({
                "action": Actions.SELECT_BOOSTER_CARD.name,
                "params": [
                    {
                        "name": "booster_card_index",
                        "type": "int",
                        "required": True,
                        "constraints": {
                            "min_value": 1,
                            "max_value": len(game_state.get('pack_cards', [])),
                            "card_source": "pack_cards"
                        }
                    }
                ]
            })
            # Skip the booster pack
            valid_actions.append({
                "action": Actions.SKIP_BOOSTER_PACK.name,
                "params": []
            })
        elif current_state in [State.PLANET_PACK, State.STANDARD_PACK, State.BUFFOON_PACK]:
            # These packs don't target hand cards
            valid_actions.append({
                "action": Actions.SELECT_BOOSTER_CARD.name,
                "params": [
                    {
                        "name": "booster_card_index",
                        "type": "int",
                        "required": True,
                        "constraints": {
                            "min_value": 1,
                            "max_value": len(game_state.get('pack_cards', [])),
                            "card_source": "pack_cards"
                        }
                    }
                ]
            })
            valid_actions.append({
                "action": Actions.SKIP_BOOSTER_PACK.name,
                "params": []
            })
        
        

        return valid_actions

    def handle_state(self, state):
        game_state_enum = State(state['state'])


        if game_state_enum in self.policy_states:
            return None

        handler = self.state_handlers.get(game_state_enum)
        if handler:
            return handler(state)

        print(f"Warning: No handler defined for state {game_state_enum.name}. Using default pass action.")
        return [Actions.PASS]

    def wait_ready_state(self):
        while not self.connected:
            try:
                self.connect_socket()
            except Exception as e:
                return False
        tries = 0
        max_tries = 100
        while tries < max_tries:
            if self.get_status() == 'READY':
                return True
            time.sleep(0.01) # this is 0.1 seconds, so 10 times per second
            tries += 1
        raise ConnectionError("Balatro did not reach READY state after 10 seconds. Is it running?")

    def run_step(self):
        if not self.connected:
            try:
                self.connect_socket()
            except Exception as e:
                return False

        try:
            status = self.get_status()
            if status == 'READY':
                self.G = self.get_state()
                max_retries = 5
                retries = 0
                while retries < max_retries:
                    is_valid = True
                    if not self.G.get('state'): # case where state is not even there
                        is_valid = False
                    else:
                        game_state_enum = State(self.G['state'])
                        if game_state_enum in EXPECTED_STATE_COMPONENTS:
                            expected_components = EXPECTED_STATE_COMPONENTS[game_state_enum]
                            for component in expected_components:
                                if component not in self.G:
                                    if self.verbose:
                                        print(f"Incomplete state for {game_state_enum.name}, missing {component}. Retrying...")
                                    is_valid = False
                                    break
                    
                    if is_valid:
                        break
                    else:
                        time.sleep(0.05) # wait a bit before retrying
                        self.G = self.get_state()
                        retries += 1
                
                if retries == max_retries:
                    raise ConnectionError("Failed to get a valid game state after 5 retries.")

                state = self.G.get('state', None)
                if not state:
                    return False # Should not happen if READY

                if self.verbose:
                    print(f"run_step: Current game state: {State(state).name}")

                for ps in self.policy_states:
                    if state == ps.value:
                        if self.verbose:
                            print(f"run_step: Current state is a policy state: {State(state).name}. Escalating to policy...")
                        return True # Escalate to policy
                if self.verbose:
                    print(f"run_step: Current state is not a policy state: {State(state).name}. Continuing...")
                
                action = self.handle_state(self.G)
                if action:
                    cmdstr = self.actionToCmd(action)
                    if self.verbose:
                        print(f"run_step: Sending action command: {cmdstr}")
                    self.sendcmd(cmdstr)
                    # We don't wait for a response here, the next loop will check status
                else:
                    if self.verbose:
                        print("run_step: No action returned by handler.")
                    pass
            else: # Status is BUSY
                time.sleep(0.01) # Wait before checking status again

        except socket.timeout:
            raise ConnectionError("Socket timed out. Is Balatro running?")
        except (socket.error, ConnectionError) as e:
            time.sleep(0.1) # Wait before trying to reconnect
            self.connected = False # Mark as disconnected
            if self.sock:
                self.sock.close()
                self.sock = None
        
        return False

    def connect_socket(self):
        if self.sock:
            self.sock.close()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(10)
        self.sock.connect(self.addr)
        self.connected = True

    def run_until_policy(self):
        escalate = False
        while not escalate:
            escalate = self.run_step()
        return self.G # Return the game state that caused the loop to stop
    
    
    def wait_for_action_result(self, timeout=10):
        """
        Wait for action result (success/failure) from the Lua side.
        Returns True for success, False for failure, None for timeout.
        """
        if not self.sock:
            return None
            
        # Save original timeout
        original_timeout = self.sock.gettimeout()
        
        try:
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.sock.settimeout(0.1)
                    data, _ = self.sock.recvfrom(65536)
                    response = json.loads(data)
                    
                    # Check if this response contains an action result
                    if 'response' in response and isinstance(response['response'], dict):
                        if 'action_result' in response['response']:
                            return response['response']['action_result']
                    elif 'action_result' in response:
                        return response['action_result']
                        
                except (socket.timeout, json.JSONDecodeError):
                    continue
                except Exception as e:
                    if self.verbose:
                        print(f"Error waiting for action result: {e}")
                    break
                    
            return None  # Timeout or error
        finally:
            # Restore original timeout
            if self.sock:
                self.sock.settimeout(original_timeout)

    def do_policy_action(self, action):
        """
        Perform a policy action on the Balatro instance.
        This method sends the action command to the Balatro instance and waits for the policy to be reached.

        args: action (list): The action to perform (e.g., [Actions.SELECT_HAND_CARD, 1] or [Actions.PLAY_SELECTED]), including the action type and any necessary arguments.
        
        returns: tuple (success: bool, game_state: dict) - success indicates if action succeeded, game_state is the resulting state
        """
        if not self.connected:
            raise ConnectionError("Not connected to Balatro instance.")
        status = self.get_status()
        if status == 'READY':
            cmdstr = self.actionToCmd(action)
            self.sendcmd(cmdstr)
            
            # Wait for action result
            action_success = self.wait_for_action_result()
            if action_success is None:
                if self.verbose:
                    print("Warning: Action result timeout, assuming failure")
                action_success = False
            
            # Get the resulting game state
            game_state = self.run_until_policy()
            return action_success, game_state
        else:
            raise ConnectionError(f"Cannot perform action, current status is {status}. Expected 'READY'.")


    def close(self):
        """Clean up all resources - safe to call multiple times"""
        try:
            # Stop the Balatro instance if it's running
            if self.balatro_instance:
                self.stop_balatro_instance()
                self.balatro_instance = None
            
            # Close the socket connection if it's open
            if self.sock:
                self.sock.close()
                self.sock = None
            
            # Clean up display resources
            if hasattr(self, 'display_num') and platform.system() == "Linux":
                self.cleanup_display(self.display_num)
            
            # Release allocated port
            if hasattr(self, 'port'):
                release_port(self.port)
            
            # Reset the bot's running state and connection status
            self.connected = False
            
            # Clear any stored state
            self.G = None
            
        except Exception:
            pass  # Ignore errors during cleanup

    def run_as_cli(self):
        print("Starting Balatro environment in CLI mode.")
        while True:
            game_state = self.run_until_policy()
            # Policy state reached, prompt user for action
            print("Current game state:", format_game_state(game_state))
            print(f"Policy required for state: {State(game_state['state']).name}")
            print("Enter action (e.g., SELECT_HAND_CARD|1 then PLAY_SELECTED, or SKIP_BLIND, or PASS to let the game continue):")
            print("Available actions:")
            actions = self.get_valid_actions(game_state)
            for action in actions:
                action_name = action['action']
                params = ", ".join([f"{p['name']} ({p['type']})" for p in action.get('params', [])])
                print(f" - {action_name}: {params if params else 'No parameters required'}")
            print("Type 'PASS' to let the game continue without an action, or 'QUIT' to exit CLI mode.")
            user_input = input("> ")
            
            if user_input.upper() == "PASS":
                print("Passing control back to the game.")
                continue # Continue the loop, letting the game proceed without sending an action
            
            if user_input.upper() == "QUIT":
                print("Exiting CLI mode.")
                self.close()
                sys.exit(0)
            # Parse user input into an action list
            try:
                action_parts = user_input.split('|')
                action_enum = Actions[action_parts[0].upper()]
                action_args = []
                if len(action_parts) > 1:
                    # Handle single integer for hand card selection
                    if action_enum == Actions.SELECT_HAND_CARD:
                        action_args.append(int(action_parts[1]))
                    else:
                        action_args.append(action_parts[1]) # For other actions, treat as string
                
                action = [action_enum] + action_args
                print(f"Executing policy action: {action}")
                success, game_state = self.do_policy_action(action)
                if success:
                    print("✓ Action succeeded")
                else:
                    print("✗ Action failed")
            except (KeyError, ValueError, IndexError) as e:
                print(f"Invalid input: {e}")

    def screenshot_np(self):
        #take a screenshot using xvfb and xwd
        if not self.display:
            raise RuntimeError("Display not set. Call setup_display_for_linux() first.")
        xwd_file = f"/tmp/balatro_screenshot_{self.display}.xwd"
        png_file = f"/tmp/balatro_screenshot_{self.display}.png"
        try:
            subprocess.run(['xwd', '-root', '-display', self.display, '-out', xwd_file], check=True)
            subprocess.run(['xwd', '-root', '-display', self.display, '-out', xwd_file], check=True)
            subprocess.run(['convert', xwd_file, png_file], check=True)
            
            #use PIL to read the PNG file
            from PIL import Image
            img = Image.open(png_file)
            img_np = np.array(img)
            img.close()
            # Clean up temporary files
            os.remove(xwd_file)
            os.remove(png_file)
            return img_np
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to take screenshot: {e}")

class BasicBalatroController(BalatroControllerBase):
    def __init__(self, verbose=False, auto_start=True):
        super().__init__(verbose=verbose, policy_states=[State.SELECTING_HAND, State.SHOP, State.BUFFOON_PACK,
                                                         State.PLANET_PACK, State.STANDARD_PACK, State.SPECTRAL_PACK, State.TAROT_PACK], auto_start=auto_start)
        # Define handlers for states that should be automated.
        self.state_handlers[State.MENU] = self.handle_menu
        self.state_handlers[State.BLIND_SELECT] = self.handle_blind_select
        self.state_handlers[State.GAME_OVER] = self.handle_game_over
        self.state_handlers[State.ROUND_EVAL] = self.handle_round_eval
        

    def handle_menu(self, state):
        """Starts a new run from the main menu."""
        return [Actions.START_RUN, 1, "Red Deck", "H8J6D1U", None]
    
    def handle_game_over(self, state):
        """go back to the main menu after the game is over."""
        return [Actions.START_RUN, 1, "Red Deck", "H8J6D1U", None]

    def handle_blind_select(self, state):
        """Automatically selects the first available blind."""
        return [Actions.SELECT_BLIND]

    def handle_round_eval(self, state):
        return [Actions.CASH_OUT]

class TrainingBalatroController(BalatroControllerBase):
    def __init__(self, verbose=False):
        # The policy is invoked for SELECTING_HAND, SHOP, and GAME_OVER states
        super().__init__(verbose=verbose, policy_states=[State.SELECTING_HAND, State.SHOP, State.GAME_OVER], auto_start=True)

        # Define handlers for states that should be automated.
        self.state_handlers[State.MENU] = self.handle_menu
        self.state_handlers[State.BLIND_SELECT] = self.handle_blind_select
        self.state_handlers[State.ROUND_EVAL] = self.handle_round_eval
        
        # Simple permissions: actions that should be blocked for RL training
        self.blocked_actions = {Actions.START_RUN, Actions.RETURN_TO_MENU, Actions.CASH_OUT}
        
        # Track episode state

    def do_policy_action(self, action):
        """Override to block restricted actions - just no-op if blocked"""
        if len(action) > 0 and action[0] in self.blocked_actions:
            if self.verbose:
                print(f"Blocked action {action[0].name} - no-op")
            # Return failed action result to indicate the action was invalid
            return False, self.G
        
        # Call parent implementation for allowed actions
        return super().do_policy_action(action)

    def handle_menu(self, state):
        """Starts a new run from the main menu."""
        return [Actions.START_RUN, 1, "Red Deck", "H8J6D1U", None]

    def handle_blind_select(self, state):
        """Automatically selects the first available blind."""
        return [Actions.SELECT_BLIND]

    def handle_booster_pack(self, state):
        """Skips any booster pack."""
        return [Actions.SKIP_BOOSTER_PACK]

    def handle_round_eval(self, state):
        return [Actions.CASH_OUT]
    
    def is_episode_done(self, game_state):
        """Check if the current episode is complete."""
        return game_state.get('state') == State.GAME_OVER.value
  
    def restart_run(self):
        self.wait_ready_state()
        self.sendcmd(self.actionToCmd(self.handle_menu(self.G)))
        return self.run_until_policy()

if __name__ == '__main__':
    env = BasicBalatroController(verbose=True)
    try:
        env.run_as_cli()
    except KeyboardInterrupt:
        print("CLI stopped by user.")
    finally:
        env.close()
