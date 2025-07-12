"""
HuggingFace Tool Calling Implementation for Balatro Agent

This module provides a clean implementation of tool calling for Balatro gameplay
using HuggingFace transformers, moving away from the art package dependency.
"""

import json
import random
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from controller import BasicBalatroController, State, Actions, format_game_state


class BalatroToolCaller:
    """
    HuggingFace-based tool calling implementation for Balatro gameplay.
    Converts Balatro actions to HuggingFace tool format and handles model interactions.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-1.7B", device: str = "auto", verbose: bool = False):
        """
        Initialize the tool caller with a HuggingFace model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ("auto", "cpu", "cuda")
        """
        self.model_name = model_name
        self.verbose = verbose
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # System prompt for Balatro gameplay
        self.system_prompt = """You are playing Balatro. There is no user - you must end each message with a valid tool call corresponding to a balatro action. Each round, you can play or discard cards from your hand until you score enough chips to beat the round, or until you are out of hands, in which case you lose. Playing cards can score chips, discarding cards lets you draw new cards from the deck. Your score is your CHIPS x MULT. Each poker hand you play has a certain number of chips and mult, with rarer hand having higher values. The hands are:
         - High Card: 5 Chips x 1 Mult
         - Pair: 10 Chips x 2 Mult
         - Two Pair: 20 Chips x 2 Mult
         - Three of a Kind: 30 Chips x 3 Mult
         - Straight: 30 Chips x 4 Mult (Aces can be high or low)
         - Flush: 35 Chips x 4 Mult
         - Full House: 40 Chips x 4 Mult
         - Four of a Kind: 60 Chips x 7 Mult
         - Straight Flush: 100 Chips x 8 Mult (Straight with Flush)
         - Five of a Kind: 150 Chips x 12 Mult (Five cards of the same rank)
         - Flush House: 140 Chips x 14 Mult (Full House with Flush)
         - Flush Five: 200 Chips x 16 Mult (Five of a Kind with Flush)
        
        It is usually better to discard cards in hopes of making a good hand rather than playing a bad hand. You may play or discard between one and five cards from your hand.

        After each round, you enter the shop phase. In the shop, you can buy cards, booster packs, or vouchers. Jokers are special cards which cannot be played, but are permanently active and provide various special abilities."""

    def actions_to_tools(self, valid_actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Balatro valid actions to HuggingFace tool calling format.
        
        Args:
            valid_actions: List of valid action specifications from balatro-controllers
            
        Returns:
            List of tool definitions in HuggingFace format
        """
        tools = []
        
        for action_spec in valid_actions:
            action_name = action_spec["action"]
            
            # Create tool definition
            tool = {
                "type": "function",
                "function": {
                    "name": action_name.lower(),
                    "description": self._get_action_description(action_name),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Convert action parameters to tool parameters
            for param in action_spec.get("params", []):
                param_name = param["name"]
                param_type = param["type"]
                
                # Convert parameter type and constraints
                if param_type == "list":
                    # Special handling for card-related parameters
                    if "card" in param_name.lower():
                        description = f"List of card indices in hand (1-based) for {param_name}"
                    else:
                        description = f"List of {param_name}"
                    
                    tool_param = {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": description
                    }
                    
                    # Add constraints
                    constraints = param.get("constraints", {})
                    if "min_length" in constraints:
                        tool_param["minItems"] = constraints["min_length"]
                    if "max_length" in constraints:
                        tool_param["maxItems"] = constraints["max_length"]
                    if "allowed_values" in constraints:
                        tool_param["items"]["enum"] = constraints["allowed_values"]
                        
                elif param_type == "int":
                    tool_param = {
                        "type": "integer",
                        "description": f"Integer value for {param_name}"
                    }
                    
                    constraints = param.get("constraints", {})
                    if "min_value" in constraints:
                        tool_param["minimum"] = constraints["min_value"]
                    if "max_value" in constraints:
                        tool_param["maximum"] = constraints["max_value"]
                        
                elif param_type == "str":
                    tool_param = {
                        "type": "string",
                        "description": f"String value for {param_name}"
                    }
                    
                    constraints = param.get("constraints", {})
                    if "allowed_values" in constraints:
                        tool_param["enum"] = constraints["allowed_values"]
                else:
                    # Default to string
                    tool_param = {
                        "type": "string",
                        "description": f"Value for {param_name}"
                    }
                
                tool["function"]["parameters"]["properties"][param_name] = tool_param
                
                # Add to required if specified
                if param.get("required", False):
                    tool["function"]["parameters"]["required"].append(param_name)
            
            tools.append(tool)
        
        return tools
    
    def _get_action_description(self, action_name: str) -> str:
        """Get human-readable description for each action."""
        descriptions = {
            "PLAY_HAND": "Play selected cards from your hand to score points. Use card indices (1-based) from your current hand.",
            "DISCARD_HAND": "Discard selected cards to draw new ones. Use card indices (1-based) from your current hand.",
            "SELECT_BLIND": "Select the current blind to continue the round",
            "SKIP_BLIND": "Skip the current blind (if possible)",
            "END_SHOP": "Leave the shop and continue to the next round",
            "REROLL_SHOP": "Spend money to reroll shop offerings",
            "BUY_CARD": "Purchase a card from the shop",
            "BUY_VOUCHER": "Purchase a voucher from the shop",
            "BUY_BOOSTER": "Purchase a booster pack from the shop",
            "SELECT_BOOSTER_CARD": "Select a card from an opened booster pack",
            "SKIP_BOOSTER_PACK": "Skip the current booster pack without selecting cards",
            "SELL_JOKER": "Sell a joker for money",
            "USE_CONSUMABLE": "Use a consumable item",
            "SELL_CONSUMABLE": "Sell a consumable item for money",
            "START_RUN": "Start a new game run",
            "RETURN_TO_MENU": "Return to the main menu",
            "CASH_OUT": "Cash out and end the current round",
            "PASS": "Take no action and continue"
        }
        return descriptions.get(action_name, f"Execute {action_name} action")
    
    def tool_call_to_action(self, tool_call: Dict[str, Any]) -> List[Any]:
        """
        Convert a tool call back to a Balatro action format.
        
        Args:
            tool_call: Tool call from model response
            
        Returns:
            Action list in format expected by balatro-controllers
        """
        function_name = tool_call["name"].upper()
        arguments = tool_call.get("arguments", {})

        # Convert function name back to Actions enum
        try:
            action_enum = Actions[function_name]
        except KeyError:
            raise ValueError(f"Unknown action: {function_name}")
        
        # Build action list
        action = [action_enum]
        
        # Add arguments based on action type
        # Note: card indices should be 1-based positions in the current hand
        if function_name == "PLAY_HAND" and "cards_to_play" in arguments:
            action.append(arguments["cards_to_play"])
        elif function_name == "DISCARD_HAND" and "cards_to_discard" in arguments:
            action.append(arguments["cards_to_discard"])
        elif function_name == "SELECT_BOOSTER_CARD":
            if "booster_card_index" in arguments:
                action.append(arguments["booster_card_index"])
            if "hand_card_indices" in arguments:
                # hand_card_indices are 1-based positions in current hand
                action.append(arguments["hand_card_indices"])
        elif function_name == "START_RUN":
            # Add optional parameters with defaults
            action.extend([
                arguments.get("stake", 1),
                arguments.get("deck", "Red Deck"),
                arguments.get("seed", None),
                arguments.get("challenge", None)
            ])
        # Add other action-specific argument handling as needed
        
        return action
    
    def generate_action(self, game_state: Dict[str, Any], valid_actions: List[Dict[str, Any]]) -> List[Any]:
        """
        Generate an action using the model with tool calling.
        
        Args:
            game_state: Current game state from balatro-controllers
            valid_actions: List of valid actions from balatro-controllers
            
        Returns:
            Action in format expected by balatro-controllers
        """
        # Format game state for model
        formatted_state = format_game_state(game_state)
        
        # Convert actions to tools
        tools = self.actions_to_tools(valid_actions)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Current game state:\n{formatted_state}\n\nChoose your next action."}
        ]
        
        # Apply chat template with tools
        text = self.tokenizer.apply_chat_template(
            messages, 
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        print(f"Model input: {text}") if self.verbose else None

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()

        response = self.tokenizer.decode(response_ids, skip_special_tokens=False)

        if self.verbose:
            print(f"Model response: {response}")

        # Parse tool calls from response
        tool_call = self._extract_tool_calls(response)

        if tool_call:
            # Convert first tool call to action
            return self.tool_call_to_action(tool_call)
        else:
            raise ValueError("No valid tool calls found in model response: " + response)
            #in the future we might want to handle this more gracefully, e.g. by falling back to a default action
            #but in dev we should fail loudly
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Extract tool calls from model response.
        This is a simplified implementation - in practice you'd want more robust parsing.
        """
        tool_calls = []
        
        # find <tool_call>...</tool_call> tags in the response
        if "<tool_call>" not in response or "</tool_call>" not in response:
            raise ValueError("No tool calls found in response: " + response)
        tool_call_str = response.split("<tool_call>")[1]
        tool_call_str = tool_call_str.split("</tool_call>")[0].strip()
        return json.loads(tool_call_str)

class BalatroGameRunner:
    """
    Main game runner that orchestrates the tool calling agent with Balatro.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the game runner.
        
        Args:
            model_name: HuggingFace model to use for tool calling
            verbose: Whether to print debug information
        """
        self.tool_caller = BalatroToolCaller(verbose=verbose)
        self.controller = BasicBalatroController(verbose=verbose)
        self.verbose = verbose
    
    def run_single_game(self) -> Dict[str, Any]:
        """
        Run a single game using tool calling.
        
        Returns:
            Game results including final score and statistics
        """
        game_state = self.controller.run_until_policy()
        actions_taken = []
        
        while game_state['state'] != State.GAME_OVER.value:
            if self.verbose:
                print(f"Current state: {State(game_state['state']).name}")
                print(format_game_state(game_state))
            
            # Get valid actions
            valid_actions = self.controller.get_valid_actions(game_state)
            
            if self.verbose:
                print(f"Available actions: {[a['action'] for a in valid_actions]}")
            
            # Generate action using tool calling
            action = self.tool_caller.generate_action(game_state, valid_actions)
            actions_taken.append(action)
            
            if self.verbose:
                print(f"Taking action: {action}")
            
            # Execute action
            game_state = self.controller.do_policy_action(action)
        
        # Game over - extract results
        final_score = game_state.get("ante", 0) - 1
        
        results = {
            "final_score": final_score,
            "actions_taken": len(actions_taken),
            "final_state": game_state
        }
        
        if self.verbose:
            print(f"Game finished! Final score (ante reached): {final_score}")
        
        return results
    
    def close(self):
        """Clean up resources."""
        self.controller.close()


def main():
    """Example usage of the tool calling system."""
    runner = BalatroGameRunner(verbose=True)
    
    try:
        results = runner.run_single_game()
        print(f"Game Results: {results}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()