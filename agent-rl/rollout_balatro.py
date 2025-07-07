import art
import time
import math
from pydantic import BaseModel
from controller import BasicBalatroController, State, Actions, format_game_state

class BalatroScenario(BaseModel):
    step: int

@art.retry()
async def rollout(
    model: art.Model, scenario: BalatroScenario
) -> art.Trajectory:
    """
    Plays a single game of Balatro, collecting the trajectory for training.
    """
    env = BasicBalatroController(verbose=True)
    game_state = env.run_until_policy() # Start the game and run until a decision is needed

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": """You are an expert Balatro player. The game consists of multiple Antes, each consisting of 3 rounds, where you must play cards to win chips. The final round is a boss fight, with special modifiers. Your goal is to reach the maximum ante possible. Success in balatro requires balancing value and tempo - you must find a good growth engine to scale your scoring potential, while also surviving in the early rounds.
                
                Each round, you must score a certain number of chips to win, or the game will end. You can play or discard cards from your hand. Playing cards can score chips, discarding cards lets you draw new cards from the deck. Your score is your CHIPS x MULT. Each poker hand you play has a certain number of chips and mult, with rarer hand having higher values. Balatro allows you to play secret hands, such as Five of a Kind, Flush Five, or Flush House, which are not normally possible in a standard 52-card deck. However, this rare hands may be somewhat inconsistent, so there is value in not greeding too hard for them. The highest ranking hand that you play is the one that is scored - so for example, if you played 4 Jack of Hearts and a Queen of Hearts, the 4 of a kind supercedes the flush, and the Queen of Hearts is not scored.

                After each round, you enter the shop phase. In the shop, you can buy cards, booster packs, or vouchers. Jokers are special cards which are permanently active and provide various special abilities. There are also consumables, which are one-time use items that can help you in the game. Consumables are most often drawn from booster packs, but can also be bought as individual cards.

                The booster packs allow you to select 1-2 out of 3-5 cards, depending on the pack size. They can also be skipped, in case none of the options are appealing. The booster pack types are as follows:
                 - Arcana Pack: Contains tarot cards, which are consumables which allow you to modify your deck.
                 - Buffoon Pack: Contains jokers.
                 - Celestial Pack: Contains planet cards, which are consumables that provide permanent bonuses to the CHIPS and MULT of different poker hands.
                 - Spectral Cards: Rare cards with powerful but potentially destructive effects.
                 - Standard Pack: Contains standard playing cards which may be added to your deck. High chance of having enhancements, seals, or editions.

                A playing card may have one enhancement, seal, and edition each. 
                The enhancements are:
                 - Bonus Card: +30 Chips
                 - Mult Card: +4 Mult
                 - Wild Card: Is considered to be every suit simultaneously.
                 - Glass Card: x2 Mult, 1 in 4 chance for card to be destroyed after use.
                 - Steel Card: x1.5 Mult while this card is in hand.
                 - Stone Card: +50 Chips, no rank or suit, card always scores when played.
                 - Gold Card: $3 if this card is held in hand at the end of the round.
                 - Lucky Card: 1 in 5 chance for +20 Mult, 1 in 15 chance to win $20.

                Seals:
                 - Gold Seal: $3 when this card is played and scored.
                 - Red Seal: Retrigger this card 1 time.
                 - Blue Seal: Creates the Planet card for the final played poker hand of the round when held in hand.
                 - Purple Seal: Creates a random tarot card when discarded.

                Only playing cards may have enhancements and seals. Jokers and playing cards may have editions. The editions are:
                 - Foil: +50 Chips when scored or Joker activated.
                 - Holographic: +10 Mult when scored or Joker activated.
                 - Polychrome: x1.5 Mult when scored or Joker activated.
                 - Negative (Joker only): +1 Joker Slot.

                Jokers activate in order from left to right.

                You should end your output with a valid action in the format: <action>CMD|ARG1|ARG2</action>, where the numbers are the indices of the cards you want to play. Lists should be formatted as comma-separated values, e.g. <action>PLAY_HAND|1,2,3</action>. If a command requires no arguments, you can simply use <action>CMD</action>. Possible actions are listed below.""",
            }
        ],
        reward=0,
    )

    while game_state['state'] != State.GAME_OVER.value:
        # 1. Format the game state for the model
        formatted_state = format_game_state(game_state)
        actions = env.get_valid_actions()
        def format_action(action):
            output = []
            output.append(action.name + " with params: ")
            for param in action.params: #params is a list of objects with field "name" (str) "type" (str) "required" (bool) and "constraints" (object)
                output.append(f"{param.name} ({param.type}) - {'required' if param.required else 'optional'}")
                for key, value in param.constraints.items():
                    output.append(f"  {key}: {value}")
            return "\n".join(output)


        possible_actions = "Your available actions are:\n" + "\n".join(
            [format_action(action) for action in actions]
        )
            
        user_prompt = f"{formatted_state}\n\n{possible_actions}"

        trajectory.messages_and_choices.append(
            {"role": "user", "content": user_prompt}
        )

        # 3. Get the model's action
        client = model.openai_client()
        chat_completion = await client.chat.completions.create(
            max_completion_tokens=128,
            messages=trajectory.messages(),
            model=model.name,
        )
        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)


        # 4. Parse the model's response to an `Actions` enum.
        #    You'll need to implement robust parsing here.
        #isolate <action> and </action> tags
        action_str = content.split("<action>")[1].split("</action>")[0]
        action = action_str.split("|")
        action_name = action[0].strip()
        action[0] = Actions[action_name]

        # 5. Apply the action to the game
        game_state = env.do_policy_action(action)

        # 6. TODO: Get error messages or feedback from the game.

        # 7. Run the game until the next policy decision
        game_state = env.run_until_policy()


    # 8. TODO: Calculate the final reward.
    #    The reward should be based on the final score, money, etc.
    #    You can access these from the final `game_state`.
    final_score = game_state.get("ante")-1 
    trajectory.reward = final_score

    env.close()
    return trajectory