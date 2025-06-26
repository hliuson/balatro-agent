# Balatro Bot API Documentation

This document outlines the API for interacting with the Balatro Bot. The API allows a client to receive game state information and send commands to control the game.

## Communication

The bot and the client communicate over UDP sockets.

- **Host:** `127.0.0.1`
- **Port:** `3000` (configurable in `config.lua`)

### Handshake

To initiate communication, the client must send a "HELLO" message to the bot. The bot will then respond with the initial game state.

## Game State

The bot sends the game state to the client in JSON format. The game state object contains all the information about the current state of the game, including cards in hand, jokers, consumables, shop items, etc.

A crucial part of the game state is the `waitingForAction` boolean.

- `waitingForAction: true`: The bot is in a stable state and ready to receive a command from the client.
- `waitingForAction: false`: The bot is busy processing a command or waiting for animations to complete. The client should not send any commands when this is false.

The game state also includes an `action_no` integer, which increments with each valid action taken. This can be used to track the sequence of actions.

## Actions

The client sends actions to the bot as a string, with parts of the command separated by `|`. The first part of the string is the action name, followed by any necessary arguments.

**Example:** `PLAY_HAND|1,3,5`

This command tells the bot to play the cards at indices 1, 3, and 5 in the hand.

### Action Reference

Here is a list of all available actions and their arguments:

| Action | Arguments | Valid States | Description |
|---|---|---|---|
| `START_RUN` | `stake` (number), `deck` (string), `seed` (string), `challenge` (string) | `MENU` | Starts a new run with the specified parameters. All arguments are optional. |
| `RETURN_TO_MENU` | - | `Any` | Returns to the main menu from any game state. |
| `SELECT_BLIND` | - | `BLIND_SELECT` | Selects the currently available blind. |
| `SKIP_BLIND` | - | `BLIND_SELECT` | Skips the currently available blind. |
| `PLAY_HAND` | `card_indices` (comma-separated list of numbers) | `SELECTING_HAND` | Plays the selected cards from the hand. `card_indices` are 1-based. |
| `DISCARD_HAND` | `card_indices` (comma-separated list of numbers) | `SELECTING_HAND` | Discards the selected cards from the hand. `card_indices` are 1-based. |
| `END_SHOP` | - | `SHOP` | Exits the shop and starts the next round. |
| `REROLL_SHOP` | - | `SHOP` | Rerolls the items in the shop. |
| `BUY_CARD` | `card_index` (number) | `SHOP` | Buys a card (Joker) from the shop. `card_index` is 1-based. |
| `BUY_VOUCHER` | `voucher_index` (number) | `SHOP` | Buys a voucher from the shop. `voucher_index` is 1-based. |
| `BUY_BOOSTER` | `booster_index` (number) | `SHOP` | Buys a booster pack from the shop. `booster_index` is 1-based. |
| `SELECT_BOOSTER_CARD` | `booster_card_index` (number), `hand_card_indices` (comma-separated list of numbers) | `TAROT_PACK`, `SPECTRAL_PACK`, `STANDARD_PACK`, `BUFFOON_PACK`, `PLANET_PACK` | Selects a card from an opened booster pack. `booster_card_index` is 1-based. `hand_card_indices` is optional and used for cards that require targeting cards in hand. |
| `SKIP_BOOSTER_PACK` | - | `TAROT_PACK`, `SPECTRAL_PACK`, `STANDARD_PACK`, `BUFFOON_PACK`, `PLANET_PACK` | Skips the current booster pack selection. |
| `SELL_JOKER` | `joker_indices` (comma-separated list of numbers) | `SHOP`, `SELECTING_HAND` | Sells the selected jokers. `joker_indices` are 1-based. |
| `USE_CONSUMABLE` | `consumable_indices` (comma-separated list of numbers), `hand_card_indices` (comma-separated list of numbers) | `SELECTING_HAND` | Uses the selected consumable cards. `consumable_indices` are 1-based. `hand_card_indices` is optional and used for consumables that require targeting cards in hand. |
| `SELL_CONSUMABLE` | `consumable_indices` (comma-separated list of numbers) | `SHOP` | Sells the selected consumable cards. `consumable_indices` are 1-based. |
| `REARRANGE_JOKERS` | `order` (comma-separated list of numbers) | `SHOP`, `SELECTING_HAND` | Rearranges the jokers in the specified order. `order` is a 1-based list of the new positions for the jokers. |
| `REARRANGE_CONSUMABLES` | `order` (comma-separated list of numbers) | `SHOP`, `SELECTING_HAND` | Rearranges the consumables in the specified order. `order` is a 1-based list of the new positions for the consumables. |
| `REARRANGE_HAND` | `order` (comma-separated list of numbers) | `SELECTING_HAND` | Rearranges the cards in hand in the specified order. `order` is a 1-based list of the new positions for the cards. |
| `PASS` | - | `Any` | Does nothing. Can be used to wait for the next game state. |

