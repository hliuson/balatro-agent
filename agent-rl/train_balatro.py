import unsloth
import art
from art.local import LocalBackend
from dotenv import load_dotenv
import random
from tqdm import tqdm
from rollout_balatro import rollout, BalatroScenario

async def main():
    load_dotenv()

    random.seed(42)

    # Declare the model
    model = art.TrainableModel(
        name="balatro-agent-v1",
        project="balatro-agent",
        base_model="Qwen/Qwen3-0.6B",
    )

    backend = LocalBackend()

    # Register the model with the local Backend
    await model.register(backend)

    for i in range(await model.get_step(), 100): # Train for 100 steps
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, BalatroScenario(step=i)) for _ in range(5) # 5 rollouts per step
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
            max_exceptions=5,
        )
        await model.delete_checkpoints()
        await model.train(
            train_groups,
            config=art.TrainConfig(learning_rate=3e-5),
            _config={"logprob_calculation_chunk_size": 8},
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())