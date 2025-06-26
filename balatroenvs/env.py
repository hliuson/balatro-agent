import sys
import json
import socket
import time
from enum import Enum
#from gamestates import cache_state
import subprocess
import random
import multiprocessing
import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except socket.error:
            return False
        
def get_available_port():
    with multiprocessing.Pool(processes=1) as pool:
        available_port = next(port for port in range(12346, 65536) if pool.apply(is_port_available, (port,)))
    return available_port

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
    NEW_ROUND = 19,

class Actions(Enum): # these enums are from the lua mod. you can add new ones but it requires changes in the lua mod as well
    SELECT_BLIND = 1
    SKIP_BLIND = 2
    PLAY_HAND = 3
    DISCARD_HAND = 4
    END_SHOP = 5
    REROLL_SHOP = 6
    BUY_CARD = 7
    BUY_VOUCHER = 8
    BUY_BOOSTER = 9
    SELECT_BOOSTER_CARD = 10
    SKIP_BOOSTER_PACK = 11
    SELL_JOKER = 12
    USE_CONSUMABLE = 13
    SELL_CONSUMABLE = 14
    REARRANGE_JOKERS = 15
    REARRANGE_CONSUMABLES = 16
    REARRANGE_HAND = 17
    PASS = 18
    START_RUN = 19
    
    RETURN_TO_MENU = 21


class BalatroEnvBase:
    def __init__(
        self,
        verbose = False,
        policy_states = [],
    ):
        self.G = None
        self.port = get_available_port()
        self.addr = ("localhost", self.port)
        self.running = False
        self.balatro_instance = None
        self.sock = None
        self.verbose = verbose
        self.policy_states = policy_states
        self.first_run = True

        # State handlers are now expected to be populated by subclasses
        self.state_handlers = {}
        #init state handlers for non-interactive states
        self.state_handlers[State.HAND_PLAYED] = self.pass_action
        self.state_handlers[State.DRAW_TO_HAND] = self.pass_action
        self.start_balatro_instance()
        time.sleep(1)
        self.connect_socket()

    def pass_action(self, state):
        """
        A no-op action that can be used to pass the turn or skip an action.
        This is useful in cases where the bot does not want to take any action.
        """
        return [Actions.PASS]

    def start_balatro_instance(self):
        balatro_exec_path = (
            r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
        )
        self.balatro_instance = subprocess.Popen(
            [balatro_exec_path, str(self.port)]
        )
        
        # Ping the server until we get a response
        max_attempts = 60
        attempt = 0
        while attempt < max_attempts:
            try:
                self.connect_socket()
                self.ping()
                data = self.sock.recv(65536)
                if data:
                    print("Connected to Balatro instance")
                    return
            except Exception as e:
                if self.verbose:
                    print(f"Attempt {attempt + 1}/{max_attempts} failed: {e}")
                time.sleep(1)
            attempt += 1
        
        raise Exception("Failed to connect to Balatro instance after multiple attempts")

        

    def stop_balatro_instance(self):
        if self.balatro_instance:
            self.balatro_instance.kill()

    def ping(self):
        cmd = "HELLO"
        msg = bytes(cmd, "utf-8")
        self.sock.sendto(msg, self.addr)

    def sendcmd(self, cmd, **kwargs):
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

    def handle_state(self, state):
        game_state_enum = State(state['state'])

        if game_state_enum == State.GAME_OVER:
            self.running = False
            return None

        if game_state_enum in self.policy_states:
            self.running = False # Stop the run loop to escalate to policy
            return None

        handler = self.state_handlers.get(game_state_enum)
        if handler:
            return handler(state)

        raise NotImplementedError(f"No handler implemented for state {game_state_enum.name} and it was not escalated to policy.")

    def wait_response(self, timeout=1):
        """
        Wait for a response from the server.
        This is a placeholder for an async implementation.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                data = self.sock.recv(65536)
                if data:
                    return json.loads(data)
            except socket.timeout:
                continue
        print("No response received within the timeout period.")
        return {}

    def run_step(self):
        if not self.running:
            print("Environment is not running. Exiting run_step.")
            return False

        self.ping()
        print("Ping sent to Balatro instance.")
        try:
            self.G = self.wait_response()
            state = self.G.get('state', None)
            if not state:
                return False
            for ps in self.policy_states:
                if state == ps.value:
                    print(f"Game state {state} is in policy states. Waiting for policy action.")
                    return True
            
            if self.G.get('waitingForAction'):
                action = self.handle_state(self.G)
                if action:
                    cmdstr = self.actionToCmd(action)
                    self.sendcmd(cmdstr)
                    data = self.wait_response()
                    print(f"Received data: {data}")
                    time.sleep(0.5)
                else:
                    raise NotImplementedError(f"No action returned from handle_state for state: {state}")
            else:
                time.sleep(0.5)


        except socket.timeout:
            print("Socket timed out. Is Balatro running?")
            self.running = False
        except socket.error as e:
            print(f"Socket error: {e}, reconnecting...")
            self.connect_socket()
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in run_step: {e}")
            self.running = False
        finally:
            time.sleep(0.5)  # Ensure we don't flood the server with requests
        return False

    def connect_socket(self):
        if self.sock is None:
            self.G = None
            self.running = True
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(10)
            self.sock.connect(self.addr)

    def run_until_policy(self):
        escalate = False
        while not escalate:
            escalate = self.run_step()
        return self.G # Return the game state that caused the loop to stop

    def do_policy_action(self, action):
        if self.G.get('waitingForAction'):
            # First, validate that the action is valid for the current state.
            # This is a crucial safeguard.
            # NOTE: We need to implement this validation logic.
            # For now, we assume the policy is correct.

            cmdstr = self.actionToCmd(action)
            self.sendcmd(cmdstr)

            # After sending the action, run until the next policy decision is needed
            return self.run_until_policy()
        else:
            print("Warning: do_policy_action called when not waiting for action. Ignoring.")
            return self.G

    def run_steps(self):
        self.running = True
        while self.running:
            self.run_step()

    def close(self):
        # Stop the Balatro instance if it's running
        if self.balatro_instance:
            self.stop_balatro_instance()
        
        # Close the socket connection if it's open
        if self.sock:
            self.sock.close()
            self.sock = None
        
        # Reset the bot's running state
        self.running = False
        
        # Clear any stored state
        self.G = None

    def run_as_cli(self):
        self.running = True
        print("Starting Balatro environment in CLI mode.")
        while self.running:
            self.run_step()
        print("Balatro environment has been stopped.")

    def handle_state(self, state):
        game_state_enum = State(state['state'])

        if game_state_enum in self.policy_states:
            return None  # Stop the run loop to escalate to policy

        handler = self.state_handlers.get(game_state_enum)

        if handler:
            return handler(state)
        else:
            if self.verbose:
                print(f"No handler for state {game_state_enum.name}, and it is not in policy_states. Waiting.")
            return None

class PlayHandEnv(BalatroEnvBase):
    """
    A Balatro environment that simplifies the game to only require policy decisions
    for playing hands. All other states are handled automatically with simple, default logic.
    """
    def __init__(self, verbose=False):
        # The policy is only invoked when the game is in the SELECTING_HAND state.
        super().__init__(verbose=verbose, policy_states=[State.SELECTING_HAND])

        # Define handlers for states that should be automated.
        self.state_handlers[State.MENU] = self.handle_menu
        self.state_handlers[State.BLIND_SELECT] = self.handle_blind_select
        self.state_handlers[State.SHOP] = self.handle_shop
        self.state_handlers[State.GAME_OVER] = self.handle_game_over

    def handle_menu(self, state):
        """Starts a new run from the main menu."""
        return [Actions.START_RUN, 1, "Red Deck", self.random_seed(), None]

    def handle_blind_select(self, state):
        """Automatically selects the first available blind."""
        return [Actions.SELECT_BLIND]

    def handle_shop(self, state):
        """Automatically ends the shopping phase without buying anything."""
        return [Actions.END_SHOP]

    def handle_booster_pack(self, state):
        """Skips any booster pack."""
        return [Actions.SKIP_BOOSTER_PACK]

    def handle_pass(self, state):
        """A general handler to pass on a state where no specific action is desired."""
        return [Actions.PASS]

    def handle_game_over(self, state):
        """Stops the environment when the game is over."""
        self.running = False
        return None

if __name__ == '__main__':
    # Example of how to run the environment
    env = PlayHandEnv(verbose=True)
    
    # The environment will run until it needs a policy decision (or the game ends).
    game_state = env.run_until_policy()
    while True:
        if game_state and game_state['state'] == State.SELECTING_HAND.value:
            print("Environment is waiting for a 'play hand' action.")
            # In a real scenario, a policy model would decide which cards to play.
            # For this example, we'll just play the first 5 cards in hand.
            action = [Actions.PLAY_HAND, [1, 2, 3, 4, 5]]
            print(f"Policy action: {action}")
            env.do_policy_action(action)
            game_state = env.run_until_policy()
