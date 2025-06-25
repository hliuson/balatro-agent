import sys
import json
import socket
import time
from enum import Enum
#from gamestates import cache_state
import subprocess
import random
import gymnasium
import multiprocessing
import socket

def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))
            return True
        except socket.error:
            return False

class State(Enum):
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

class Actions(Enum):
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
    SEND_GAMESTATE = 20
    RETURN_TO_MENU = 21


class BalatroEnvBase:
    def __init__(self, policy_states):
        policy_states = policy_states
        self.port = self.get_available_port()
        self.bot = BalatroBotBase(deck="Red Deck", stake=1, seed=None, challenge=None,
                                  bot_port=self.port, verbose=True, exposed_to_policy=policy_states)
        self.bot.start_balatro_instance()
        time.sleep(1)
        self.bot.connect_socket()

    def get_available_port(self):
        with multiprocessing.Pool(processes=1) as pool:
            available_port = next(port for port in range(12346, 65536) if pool.apply(is_port_available, (port,)))
        return available_port
    
    def _step(self, action):
        # Send the action to the bot
        self.bot.do_policy_action(action)
        
        return self.bot.G      

    def _reset(self):
        # Reset the bot
        self.bot.start_run()
        return self.bot.G
        
    def close(self):
        # Stop the Balatro instance if it's running
        if self.bot.balatro_instance:
            self.bot.stop_balatro_instance()
        
        # Close the socket connection if it's open
        if self.bot.sock:
            self.bot.sock.close()
            self.bot.sock = None
        
        # Reset the bot's running state
        self.bot.running = False
        
        # Clear any stored state
        self.bot.G = None
        self.bot.state = {}
    
    def is_run_finished(self):
        return self.bot.G["waitingFor"] == "return_to_menu"
    
    def run_as_cli(self):
        while True:
            res = self.bot.run_steps()
            print(res)
            print(self.bot.G["waitingFor"])
            userinput = input("Enter action as JSON with the format: [Action, [params]] or 'exit' to quit: ")
            if userinput.lower() == 'exit':
                print("Exiting CLI...")
                break
            try:
                parsed = json.loads(userinput)
                print(f"Parsed action: {parsed}")
                self.bot.do_policy_action(parsed)
            except json.JSONDecodeError as e:
                print(f"Invalid input format: {e}")

        self.close()


class BalatroBotBase:
    def __init__(
        self,
        deck: str,
        stake: int = 1,
        seed: str = None,
        challenge: str = None,
        bot_port: int = 12346,
        verbose = False,
        exposed_to_policy = [],
    ):
        self.G = None
        self.deck = deck
        self.stake = stake
        self.seed = seed
        self.challenge = challenge

        self.bot_port = bot_port

        self.addr = ("localhost", self.bot_port)
        self.running = False
        self.balatro_instance = None

        self.sock = None

        self.state = {}
        self.verbose = verbose
        self.exposed_to_policy = exposed_to_policy
        self.first_run = True
        self.last_action_no = -1
        self.last_action_timestamp = time.time()
        self.last_cmd = "Hello"

    def skip_or_select_blind(self, G):
        return [Actions.SELECT_BLIND]

    def select_shop_action(self, G):
        return [Actions.END_SHOP]

    def select_booster_action(self, G):
        return [Actions.SKIP_BOOSTER_PACK]

    def sell_jokers(self, G):
        return [Actions.SELL_JOKER, []]

    def rearrange_jokers(self, G):
        return [Actions.REARRANGE_JOKERS, []]

    def use_or_sell_consumables(self, G):
        return [Actions.USE_CONSUMABLE, []]

    def rearrange_consumables(self, G):
        return [Actions.REARRANGE_CONSUMABLES, []]

    def rearrange_hand(self, G):
        return [Actions.REARRANGE_HAND, [1,2,4,3,5,6,7,8]]
    
    def select_cards_from_hand(self, G):
        return [Actions.PLAY_HAND, [1]]


    def start_balatro_instance(self):
        balatro_exec_path = (
            r"C:\Program Files (x86)\Steam\steamapps\common\Balatro\Balatro.exe"
        )
        self.balatro_instance = subprocess.Popen(
            [balatro_exec_path, str(self.bot_port)]
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
                print(f"Waiting for Balatro to start... (attempt {attempt + 1}/{max_attempts})")
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
        self.last_action_no += 1
        self.last_action_timestamp = time.time()
        self.last_action = cmd

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

    def verifyimplemented(self):
        try:
            self.skip_or_select_blind({})
            self.select_cards_from_hand({})
            self.select_shop_action({})
            self.select_booster_action({})
            self.sell_jokers({})
            self.rearrange_jokers({})
            self.use_or_sell_consumables({})
            self.rearrange_consumables({})
            self.rearrange_hand({})
        except NotImplementedError as e:
            print(e)
            sys.exit(0)
        except:
            pass

    def random_seed(self):
        # e.g. 1OGB5WO
        return "".join(random.choices("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=7))

    def chooseaction(self):
        if "state" in self.G:
            if self.G["state"] == State.GAME_OVER:
                self.running = False

        match self.G["waitingFor"]:
            case "start_run":
                seed = self.seed
                if seed is None:
                    seed = self.random_seed()
                return [
                    Actions.START_RUN,
                    self.stake,
                    self.deck,
                    seed,
                    self.challenge,
                ]
            case "skip_or_select_blind":
                return self.skip_or_select_blind(self.G)
            case "select_cards_from_hand":
                return self.select_cards_from_hand(self.G)
            case "select_shop_action":
                return self.select_shop_action(self.G)
            case "select_booster_action":
                return self.select_booster_action(self.G)
            case "sell_jokers":
                return self.sell_jokers(self.G)
            case "rearrange_jokers":
                return self.rearrange_jokers(self.G)
            case "use_or_sell_consumables":
                return self.use_or_sell_consumables(self.G)
            case "rearrange_consumables":
                return self.rearrange_consumables(self.G)
            case "rearrange_hand":
                return self.rearrange_hand(self.G)
            case "return_to_menu":
                return [Actions.RETURN_TO_MENU]
        raise ValueError(f"State {self.G['waitingFor']} not implemented in chooseaction or escalated to policy")
            

    def run_step(self):
        if self.running:
            self.ping()
            jsondata = {}
            try:
                data = self.sock.recv(65536)
                jsondata = json.loads(data)

                if "response" in jsondata:
                    print(jsondata["response"])
                    return False # error
                else:
                    self.G = jsondata
                    #print(self.G)
                    if self.G["waitingForAction"]:
                        if self.verbose:
                            print(f"Waiting for action: {self.G['waitingFor']}")
                        if self.G["waitingFor"] in self.exposed_to_policy:
                            self.running = False
                            return True # Escalate to Policy
                        else:
                            action = self.chooseaction()
                            if action == None:
                                raise ValueError("All actions must return a value!")

                            cmdstr = self.actionToCmd(action)
                            self.sendcmd(cmdstr)
                            print(f"Action sent: {cmdstr}")
                    else:
                        #if time.time() - self.last_action_timestamp > 1:
                        #    #reissue the last action if no response in 2 seconds
                        #    self.last_action_no -= 1
                        #    cmdstr = self.last_action
                        #    self.sendcmd(cmdstr)

                        # sleep for a bit to avoid busy waiting
                        time.sleep(0.1)
                        if self.verbose:
                            pass
            except socket.error as e:
                print(e)
                print("Socket error, reconnecting...")
                self.connect_socket()
            except Exception as e:
                raise

    def connect_socket(self):
        if self.sock is None:
            self.verifyimplemented()
            self.state = {}
            self.G = None

            self.running = True
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.settimeout(10)
            self.sock.connect(self.addr)

    def do_policy_action(self, action):
        max_attempts = 3
        attempt = 0
        while attempt < max_attempts:
            try:
                actionname = action[0].name if isinstance(action[0], Actions) else str(action[0])
                #print(f"Executing action: {actionname} (Attempt {attempt + 1}/{max_attempts})")
                self.connect_socket()
                cmdstr = self.actionToCmd(action)
                self.sendcmd(cmdstr)
                self.run_steps()
                break  # If successful, exit the loop
            except Exception as e:
                print(f"Error executing policy action: {e}")
                attempt += 1
                if attempt < max_attempts:
                    print(f"Retrying... (Attempt {attempt + 1}/{max_attempts})")
                    time.sleep(1)  # Wait a bit before retrying
                else:
                    #print("Max attempts reached. Failed to execute action.")
                    self.running = False

    def start_run(self):
        if self.first_run:
            self.first_run = False
            self.run_steps()
        else:
            action = [Actions.RETURN_TO_MENU]
            self.do_policy_action(action)

    def run_steps(self):
        self.running = True
        while self.running:
            self.run_step()

if __name__ == "__main__":
    env = BalatroEnvBase(policy_states=[])
    print(env._reset())
    env.run_as_cli()