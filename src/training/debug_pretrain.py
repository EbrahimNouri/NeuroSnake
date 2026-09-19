import torch
import torch.nn.functional as F
from src.brain.fly_brain import FlyBrain
from src.config import DEVICE

print("\nTesting FlyBrain on", DEVICE)
brain = FlyBrain().to(DEVICE)
optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

state = game.get_observation()
state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
target = torch.tensor([0], device=DEVICE, dtype=torch.long)

t0 = time.time()
for i in range(100):
    q = brain(state_t)
    loss = F.cross_entropy(q, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"100 forward+backward: {time.time()-t0:.2f}s")
print(f"avg per step: {(time.time()-t0)/100*1000:.2f} ms")