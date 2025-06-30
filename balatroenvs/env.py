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
    NEW_ROUND = 19

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
    RETURN_TO_MENU = 20
    CASH_OUT = 21


class BalatroEnvBase:
    def __init__(
        self,
        verbose = False,
        policy_states = [],
    ):
        self.G = None
        self.port = get_available_port()
        self.addr = ("localhost", self.port)
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
        self.state_handlers[State.NEW_ROUND] = self.pass_action
        self.connected = False
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
                time.sleep(0.1)
        if not data:
            raise ConnectionError("No response from Balatro instance.")
        data = json.loads(data)
        print(f"Received data: {data}") if self.verbose else None
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

    def start_balatro_instance(self):
        # Get Balatro executable path from environment variable
        balatro_exec_path = os.getenv('BALATRO_EXEC_PATH')
        
        if not balatro_exec_path:
            # Fallback to default paths based on platform
            if platform.system() == "Windows":
                balatro_exec_path = r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
            elif platform.system() == "Linux":
                balatro_exec_path = os.path.expanduser("~/.steam/steam/steamapps/common/Balatro/Balatro")
            elif platform.system() == "Darwin":  # macOS
                balatro_exec_path = os.path.expanduser("~/Library/Application Support/Steam/steamapps/common/Balatro/Balatro.app/Contents/MacOS/Balatro")
            else:
                raise Exception(f"Unsupported platform: {platform.system()}")
        
        # Check if the executable exists
        if not os.path.exists(balatro_exec_path):
            raise FileNotFoundError(f"Balatro executable not found at: {balatro_exec_path}")
        
        # Build the command
        cmd = [balatro_exec_path, str(self.port)]
        
        # On Linux, check if we need to use xvfb for headless operation
        if platform.system() == "Linux":
            # Check if DISPLAY is set, if not, use xvfb
            if not os.getenv('DISPLAY'):
                if self.verbose:
                    print("No DISPLAY environment variable found. Using xvfb for headless operation.")
                # Check if xvfb-run is available
                try:
                    subprocess.run(['which', 'xvfb-run'], check=True, capture_output=True)
                    cmd = ['xvfb-run', '-a', '-s', '-screen 0 1024x768x24'] + cmd
                except subprocess.CalledProcessError:
                    print("Warning: xvfb-run not found. Install xvfb package for headless operation.")
                    print("On Ubuntu/Debian: sudo apt-get install xvfb")
                    print("On RHEL/CentOS: sudo yum install xorg-x11-server-Xvfb")
        
        if self.verbose:
            print(f"Starting Balatro with command: {' '.join(cmd)}")
        
        self.balatro_instance = subprocess.Popen(cmd)
        if self.verbose:
            print(f"Balatro instance started with PID: {self.balatro_instance.pid} on port {self.port}")
        
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
                time.sleep(1)
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
        #return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))
        return "H8J6D1U"

    def handle_state(self, state):
        game_state_enum = State(state['state'])


        if game_state_enum in self.policy_states:
            return None

        handler = self.state_handlers.get(game_state_enum)
        if handler:
            return handler(state)

        raise NotImplementedError(f"No handler implemented for state {game_state_enum.name} and it was not escalated to policy.")

    

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
                state = self.G.get('state', None)
                if not state:
                    return False # Should not happen if READY

                if self.verbose:
                    print(f"run_step: Current game state: {State(state).name}")

                for ps in self.policy_states:
                    if state == ps.value:
                        return True # Escalate to policy
                
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
                time.sleep(1) # Wait before checking status again

        except socket.timeout:
            raise ConnectionError("Socket timed out. Is Balatro running?")
        except (socket.error, ConnectionError) as e:
            time.sleep(1) # Wait before trying to reconnect
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

    def do_policy_action(self, action):
        if not self.connected:
            if self.verbose:
                print("do_policy_action: Not connected. Cannot perform policy action.")
            return self.G
        try:
            if self.verbose:
                print("do_policy_action: Getting status before sending action.")
            status = self.get_status()
            if self.verbose:
                print(f"do_policy_action: Current status: {status}")
            if status == 'READY':
                cmdstr = self.actionToCmd(action)
                if self.verbose:
                    print(f"do_policy_action: Sending policy action command: {cmdstr}")
                self.sendcmd(cmdstr)
                if self.verbose:
                    print("do_policy_action: Policy action sent. Resuming automated steps.")
                return self.run_until_policy()
            else:
                if self.verbose:
                    print("do_policy_action: Called when bot was BUSY. Ignoring.")
                return self.G
        except ConnectionError:
            if self.verbose:
                print("do_policy_action: Connection lost during policy action. Cannot perform action.")
            return self.G

    def close(self):
        # Stop the Balatro instance if it's running
        if self.balatro_instance:
            self.stop_balatro_instance()
        
        # Close the socket connection if it's open
        if self.sock:
            self.sock.close()
            self.sock = None
        
        # Reset the bot's running state and connection status
        self.connected = False
        
        # Clear any stored state
        self.G = None

    def run_as_cli(self):
        print("Starting Balatro environment in CLI mode.")
        while True:
            game_state = self.run_until_policy()
            # Policy state reached, prompt user for action
            print(f"Policy required for state: {State(game_state['state']).name}")
            print("Enter action (e.g., PLAY_HAND|1,2,3,4,5 or SKIP_BLIND, or PASS to let the game continue):")
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
                    # Handle comma-separated list for card indices
                    if action_enum == Actions.PLAY_HAND or action_enum == Actions.DISCARD_HAND:
                        action_args.append([int(x) for x in action_parts[1].split(',')])
                    else:
                        action_args.append(action_parts[1]) # For other actions, treat as string
                
                action = [action_enum] + action_args
                print(f"Executing policy action: {action}")
                self.do_policy_action(action)
            except Exception as e:
                print(f"Invalid input or error executing action: {e}")
                print("Please try again.")

class BasicBalatro(BalatroEnvBase):
    def __init__(self, verbose=False):
        # The policy is only invoked when the game is in the SELECTING_HAND state.
        super().__init__(verbose=verbose, policy_states=[State.SELECTING_HAND, State.SHOP])

        # Define handlers for states that should be automated.
        self.state_handlers[State.MENU] = self.handle_menu
        self.state_handlers[State.BLIND_SELECT] = self.handle_blind_select
        self.state_handlers[State.GAME_OVER] = self.handle_game_over
        self.state_handlers[State.ROUND_EVAL] = self.handle_round_eval
        

    def handle_menu(self, state):
        """Starts a new run from the main menu."""
        return [Actions.START_RUN, 1, "Red Deck", self.random_seed(), None]

    def handle_blind_select(self, state):
        """Automatically selects the first available blind."""
        return [Actions.SELECT_BLIND]

    def handle_booster_pack(self, state):
        """Skips any booster pack."""
        return [Actions.SKIP_BOOSTER_PACK]

    def handle_pass(self, state):
        """A general handler to pass on a state where no specific action is desired."""
        return [Actions.PASS]

    def handle_game_over(self, state):
        """go back to the main menu after the game is over."""
        return [Actions.START_RUN, 1, "Red Deck", self.random_seed(), None]
    
    def handle_round_eval(self, state):
        return [Actions.CASH_OUT]

if __name__ == '__main__':
    env = BasicBalatro(verbose=True)
    try:
        env.run_as_cli()
    except KeyboardInterrupt:
        print("CLI stopped by user.")
    finally:
        env.close()
