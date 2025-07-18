#!/usr/bin/env python3

import time
from controller import BasicBalatroController, State, Actions

class BenchmarkController(BasicBalatroController):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.step_count = 0
        self.start_time = None
        self.first_round = True
        self.bought_shop_card = False
        self.selected_cards = []  # Track selected cards
        
    def reset_run(self):
        """Reset state for a new run."""
        self.step_count = 0
        self.start_time = None
        self.first_round = True
        self.bought_shop_card = False
        self.selected_cards = []
        
    def handle_menu(self, state):
        """Starts a new run with seed H8J6D1U."""
        if self.start_time is None:
            self.start_time = time.time()
        return [Actions.START_RUN, 1, "Red Deck", "H8J6D1U", None]
    
    def handle_selecting_hand(self, state):
        """Handle hand selection based on benchmark strategy."""
        self.step_count += 1
        
        hand = state.get('hand', [])
        round_info = state.get('round', {})
        discards_left = round_info.get('discards_left', 0)
        
        # Check if we have selected cards
        selected = [card for card in hand if card.get('highlighted', False)]
        
        if self.first_round:
            # First round: select and play first 5 cards
            if len(selected) == 0:
                # No cards selected yet, start selecting first 5
                return [Actions.SELECT_HAND_CARD, 1]
            elif len(selected) < 5 and len(selected) < len(hand):
                # Continue selecting up to 5 cards
                return [Actions.SELECT_HAND_CARD, len(selected) + 1]
            else:
                # Have 5 cards selected, play them
                self.first_round = False
                return [Actions.PLAY_SELECTED]
        else:
            # Subsequent rounds: discard down to 0, then play high cards
            if discards_left > 0:
                # Discard strategy: select one card and discard
                if len(selected) == 0:
                    # Select first card to discard
                    return [Actions.SELECT_HAND_CARD, 1]
                else:
                    # Discard selected card
                    return [Actions.DISCARD_SELECTED]
            else:
                # No discards left, play high cards
                # Select highest value cards (assume they're at the end)
                if len(selected) == 0 and hand:
                    # Select the last card (assuming it's high)
                    return [Actions.SELECT_HAND_CARD, len(hand)]
                elif len(selected) > 0:
                    # Play selected cards
                    return [Actions.PLAY_SELECTED]
        
        return [Actions.PASS]
    
    def handle_shop(self, state):
        """Handle shop: buy second card on first visit, then skip."""
        self.step_count += 1
        
        if not self.bought_shop_card:
            shop_jokers = state.get('shop', {}).get('jokers', [])
            if len(shop_jokers) >= 2:
                self.bought_shop_card = True
                return [Actions.BUY_CARD, 2]  # Buy second card
        
        return [Actions.END_SHOP]
    
    def handle_game_over(self, state):
        """Handle game over - restart the game."""
        return [Actions.START_RUN, 1, "Red Deck", "H8J6D1U", None]

def run_single_game(controller):
    """Run a single game and return (steps, time, ante)."""
    controller.reset_run()
    
    # Override state handlers for benchmark strategy
    controller.state_handlers[State.SELECTING_HAND] = controller.handle_selecting_hand
    controller.state_handlers[State.SHOP] = controller.handle_shop
    controller.state_handlers[State.GAME_OVER] = controller.handle_game_over
    
    game_over_count = 0
    max_game_overs = 2  # Avoid infinite loops
    
    while game_over_count < max_game_overs:
        game_state = controller.run_until_policy()
        current_state = State(game_state['state'])
        
        if current_state == State.GAME_OVER:
            end_time = time.time()
            elapsed_time = end_time - controller.start_time if controller.start_time else 0
            ante = game_state.get('game', {}).get('ante', 1)
            
            if game_over_count == 0:
                # First game over - this is our result
                result = (controller.step_count, elapsed_time, ante)
                game_over_count += 1
                
                # Restart for next iteration
                action = controller.handle_game_over(game_state)
                if action:
                    controller.do_policy_action(action)
                    controller.reset_run()
                return result
            else:
                game_over_count += 1
                
        elif current_state == State.SELECTING_HAND:
            action = controller.handle_selecting_hand(game_state)
            if action and action != [Actions.PASS]:
                controller.do_policy_action(action)
        elif current_state == State.SHOP:
            action = controller.handle_shop(game_state)
            if action:
                controller.do_policy_action(action)
        else:
            # Let automated handlers deal with other states
            controller.run_step()
    
    # Fallback if we hit max game overs
    return (controller.step_count, 0, 1)

def run_benchmark(num_runs=5):
    """Run the Balatro benchmark multiple times and calculate average."""
    print("Starting Balatro benchmark...")
    print("Strategy: Play first 5 cards, buy second shop card, then discard/play high cards")
    print(f"Seed: H8J6D1U")
    print(f"Number of runs: {num_runs}")
    print()
    
    controller = BenchmarkController(verbose=False)  # Less verbose for multiple runs
    
    try:
        results = []
        
        for run in range(num_runs):
            print(f"Running iteration {run + 1}/{num_runs}...")
            steps, elapsed_time, ante = run_single_game(controller)
            steps_per_second = steps / elapsed_time if elapsed_time > 0 else 0
            
            results.append({
                'steps': steps,
                'time': elapsed_time,
                'sps': steps_per_second,
                'ante': ante
            })
            
            print(f"  Steps: {steps}, Time: {elapsed_time:.2f}s, SPS: {steps_per_second:.2f}, Ante: {ante}")
        
        # Calculate averages
        avg_steps = sum(r['steps'] for r in results) / len(results)
        avg_time = sum(r['time'] for r in results) / len(results)
        avg_sps = sum(r['sps'] for r in results) / len(results)
        avg_ante = sum(r['ante'] for r in results) / len(results)
        
        print(f"\n=== BENCHMARK RESULTS ===")
        print(f"Number of runs: {num_runs}")
        print(f"Average steps: {avg_steps:.1f}")
        print(f"Average time: {avg_time:.2f} seconds")
        print(f"Average steps per second: {avg_sps:.2f}")
        print(f"Average ante reached: {avg_ante:.1f}")
        print(f"=========================\n")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        controller.close()

if __name__ == '__main__':
    import sys
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_benchmark(num_runs)