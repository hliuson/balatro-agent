#!/usr/bin/env python3

import time
import concurrent.futures
import threading
from controller import BasicBalatroController, State, Actions

# Lock to prevent multiple instances from conflicting over Lovely dump directory during boot
_dump_boot_lock = threading.Lock()

class BenchmarkController(BasicBalatroController):
    def __init__(self, verbose=False, auto_start=True):
        super().__init__(verbose=verbose, auto_start=auto_start)
        self.step_count = 0
        self.start_time = None
        self.first_round = True
        self.bought_shop_card = False
        self.selected_cards = []  # Track selected cards
        self.shop_visit_count = 0  # Track number of shop visits
        self.second_shop_actions = 0  # Track actions in second shop
        self.policy_states = [
            State.SELECTING_HAND,
            State.SHOP,
            State.GAME_OVER,
            State.SPECTRAL_PACK,
        ]  # States we handle with our policy
        
    def reset_run(self):
        """Reset state for a new run."""
        self.step_count = 0
        self.start_time = None
        self.first_round = True
        self.bought_shop_card = False
        self.selected_cards = []
        self.shop_actions = 0
        self.spectral_pack_actions = 0

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
        """Handle shop: first shop buy second card, second shop perform specific sequence."""
        self.shop_actions += 1
        
        if self.shop_actions == 1:
            return [Actions.BUY_CARD, 2]  # Buy second card
        elif self.shop_actions == 2:
            return [Actions.END_SHOP]
        elif self.shop_actions == 3:
            return [Actions.END_SHOP]
            #return [Actions.BUY_BOOSTER, 2]  # Buy second booster pack
        elif self.shop_actions == 4:
                return [Actions.END_SHOP]
        # Default: exit shop
        return [Actions.END_SHOP]

    def handle_spectral_pack(self, state):
        """Handle spectral pack - select first card."""
        print("Spectral Pack Action")
        self.spectral_pack_actions += 1

        if self.spectral_pack_actions == 1:
            return [Actions.SELECT_HAND_CARD, 1]
        elif self.spectral_pack_actions == 2:
            return [Actions.SELECT_BOOSTER_CARD, 1]
        else:
            return [Actions.SKIP_BOOSTER_PACK]

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
    controller.state_handlers[State.SPECTRAL_PACK] = controller.handle_spectral_pack
    
    # Start timing for this game
    start_time = time.time()
    
    game_over_count = 0
    max_game_overs = 2  # Avoid infinite loops
    
    while game_over_count < max_game_overs:
        game_state = controller.run_until_policy()
        current_state = State(game_state['state'])
        
        if current_state == State.GAME_OVER:
            end_time = time.time()
            elapsed_time = end_time - start_time
            ante = game_state.get('game', {}).get('ante', 1)
            
            if game_over_count == 0:
                # First game over - this is our result
                result = (controller.step_count, elapsed_time, ante)
                game_over_count += 1
                
                # Restart for next iteration
                action = controller.handle_game_over(game_state)
                if action:
                    success, game_state = controller.do_policy_action(action)
                    controller.reset_run()
                return result
            else:
                game_over_count += 1
                
        elif current_state == State.SELECTING_HAND:
            action = controller.handle_selecting_hand(game_state)
            if action and action != [Actions.PASS]:
                success, game_state = controller.do_policy_action(action)
        elif current_state == State.SHOP:
            action = controller.handle_shop(game_state)
            if action:
                success, game_state = controller.do_policy_action(action)
        elif current_state == State.SPECTRAL_PACK:
            action = controller.handle_spectral_pack(game_state)
            if action:
                success, game_state = controller.do_policy_action(action)
    
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
        print(f"Average steps per run: {avg_steps:.2f}")
        print(f"Average time per run: {avg_time:.2f}s")
        print(f"Average steps per second: {avg_sps:.2f}")
        print("=========================")
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")
    finally:
        controller.close()

def run_parallel_benchmark(num_runs=5, num_workers=4):
    """Run the Balatro benchmark using parallel processing."""
    print("Starting Parallel Balatro benchmark...")
    print("Strategy: Play first 5 cards, buy second shop card, then discard/play high cards")
    print(f"Seed: H8J6D1U")
    print(f"Number of runs: {num_runs}")
    print(f"Number of workers: {num_workers}")
    print()
    
    def run_worker_batch(games_per_worker):
        """Worker function that creates one controller and runs multiple games."""
        results = []
        
        # Create controller without starting Balatro yet
        controller = BenchmarkController(verbose=False, auto_start=False)
        
        # Acquire lock to prevent Lovely dump directory conflicts during boot
        with _dump_boot_lock:
            # Start Balatro instance while holding the lock
            controller.start_balatro_instance()
        
        # Now the game is booted, run multiple games with this instance
        try:
            for _ in range(games_per_worker):
                result = run_single_game(controller)
                results.append(result)
            return results
        finally:
            controller.close()
    
    try:
        start_time = time.time()
        
        # Distribute games across workers
        games_per_worker = num_runs // num_workers
        remaining_games = num_runs % num_workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            print(f"Distributing {num_runs} games across {num_workers} workers...")
            
            # Submit worker batches
            futures = []
            for i in range(num_workers):
                # Give extra games to first few workers if there's a remainder
                worker_games = games_per_worker + (1 if i < remaining_games else 0)
                if worker_games > 0:
                    futures.append(executor.submit(run_worker_batch, worker_games))
            
            # Collect results as they complete
            results = []
            completed_runs = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    worker_results = future.result()
                    for steps, elapsed_time, ante in worker_results:
                        steps_per_second = steps / elapsed_time if elapsed_time > 0 else 0
                        
                        results.append({
                            'steps': steps,
                            'time': elapsed_time,
                            'sps': steps_per_second,
                            'ante': ante
                        })
                        
                        completed_runs += 1
                        print(f"  Run {completed_runs} completed: Steps: {steps}, Time: {elapsed_time:.2f}s, SPS: {steps_per_second:.2f}, Ante: {ante}")
                        
                except Exception as e:
                    print(f"  Worker batch failed: {e}")
        
        total_time = time.time() - start_time
        
        if results:
            # Calculate averages
            avg_steps = sum(r['steps'] for r in results) / len(results)
            avg_time = sum(r['time'] for r in results) / len(results)
            avg_sps = sum(r['sps'] for r in results) / len(results)
            avg_ante = sum(r['ante'] for r in results) / len(results)
            
            print(f"\n=== PARALLEL BENCHMARK RESULTS ===")
            print(f"Number of runs: {len(results)}/{num_runs}")
            print(f"Number of workers: {num_workers}")
            print(f"Total wall time: {total_time:.2f}s")
            print(f"Average steps per run: {avg_steps:.2f}")
            print(f"Average time per run: {avg_time:.2f}s")
            print(f"Average steps per second: {avg_sps:.2f}")
            print(f"Average ante reached: {avg_ante:.2f}")
            print(f"Speedup factor: {(avg_time * len(results)) / total_time:.2f}x")
            print("===================================")
        else:
            print("No successful runs completed.")
            
    except KeyboardInterrupt:
        print("\nParallel benchmark interrupted by user.")

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == '--parallel':
        num_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        run_parallel_benchmark(num_runs, num_workers)
    else:
        num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
        run_benchmark(num_runs)